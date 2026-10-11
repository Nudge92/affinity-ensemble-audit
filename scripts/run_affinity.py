"""
run_affinity.py — Affinity 예측 스크립트
==============================================
Snakemake rule 'affinity'에서 호출됩니다.

모델 목록:
  1. CheapNet_Cross   → conda env: CheapNet_Cross   (GNN, 기존)
  2. CheapNet_Diverse → conda env: CheapNet_Diverse  (GNN, 기존)
  3. PIGNet2          → conda env: pignet2           (Physics-informed GNN, 신규)
  4. OnionNet2        → conda env: onionnet2         (2D CNN, 신규)
  5. DeepDTA          → conda env: deepdta           (1D sequence CNN, 신규)

입력: results/02_admet/{gen}-{admet}-filtered.csv + data/target_protein.pdb
출력: results/03_affinity/{gen}-{admet}-{aff}-scores.csv
      (컬럼: ID, SMILES, Affinity_Score, Affinity_Model)

로그 구조:
  logs/03_affinity/{gen}-{admet}-{aff}.log
  logs/03_affinity/errors/{gen}-{admet}-{aff}-errors.jsonl

변경 이력:
  2026-04-03: SMINA docking 추가 (PIGNet2, OnionNet2)
              DeepDTA 단백질 서열 β2AR(4LDE)로 수정
  2026-04-04: CDK2(1AQ1) 타겟으로 전환
              POCKET_CENTER, TARGET_SEQUENCE 변경
"""
import logging
import traceback
import subprocess
import os
import json
import tempfile
from datetime import datetime

import pandas as pd

# ─── Snakemake 인터페이스 ───
log_file     = snakemake.log[0]
ligands_file = snakemake.input.ligands
protein_file = snakemake.input.protein
output_file  = snakemake.output[0]
model_name   = snakemake.params.model_name

# ─── conda 환경별 Python 경로 ───
CONDA_PYTHON = {
    "PIGNet2":   "conda run -n pignet2 python",
    "OnionNet2": "conda run -n onionnet2 python",
    "DeepDTA":   "conda run -n deepdta python",
    "GraphDTA":  "conda run -n pignet2 python",
    "MGraphDTA": "conda run -n pignet2 python",
    "GIGN":      "conda run -n pignet2 python",
    "PSICHIC":   "conda run -n psichic_fp python",
    "NMDN":      "conda run -n nmdn python",
}

# ─── SMINA 설정 ───
SMINA_BIN = os.path.expanduser("~/bin/smina")

# Binding pocket 중심 좌표 (Pocket2Mol 생성 시 사용한 좌표)
# 타겟: CDK2 (PDB 1AQ1, co-crystal ligand STU 중심)
POCKET_CENTER = (0.5, 27.1, 9.0)
POCKET_BOX_SIZE = 20.0  # Å (각 축 ±10Å)

# ═══════════════════════════════════════════════════════════════
# 로깅 설정
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

error_log_dir = os.path.join(os.path.dirname(log_file), "errors")
os.makedirs(error_log_dir, exist_ok=True)
error_log_path = os.path.join(
    error_log_dir,
    os.path.basename(log_file).replace(".log", "-errors.jsonl")
)


def log_smiles_error(smiles: str, error_msg: str, model: str):
    """개별 SMILES 에러를 JSONL에 기록"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "smiles": smiles,
        "error": error_msg,
    }
    with open(error_log_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_smiles_column(df: pd.DataFrame) -> str:
    """SMILES 컬럼 자동 탐색"""
    for col in df.columns:
        if col.upper() == "SMILES":
            return col
    raise ValueError(f"SMILES 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════
# 공통 유틸리티: SMINA Docking
# ═══════════════════════════════════════════════════════════════

def smina_dock_sdf(input_sdf: str, protein_pdb: str, output_sdf: str,
                   center: tuple, box_size: float = 20.0,
                   exhaustiveness: int = 8, num_modes: int = 1) -> bool:
    """
    SMINA로 docking 수행.
    input_sdf의 모든 분자를 protein_pdb에 docking하여 output_sdf에 저장.

    Args:
        input_sdf: 입력 리간드 SDF (3D conformer)
        protein_pdb: 단백질 PDB
        output_sdf: docking 결과 SDF (best pose)
        center: (x, y, z) pocket 중심 좌표
        box_size: search box 크기 (Å)
        exhaustiveness: 탐색 강도 (높을수록 정확하지만 느림)
        num_modes: 반환할 pose 수

    Returns:
        True if successful
    """
    cmd = [
        SMINA_BIN,
        "--receptor", protein_pdb,
        "--ligand", input_sdf,
        "--out", output_sdf,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size),
        "--size_y", str(box_size),
        "--size_z", str(box_size),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--quiet",
    ]
    logging.info(f"SMINA docking: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        logging.warning(f"SMINA 실패: {r.stderr[:500]}")
        return False

    if not os.path.exists(output_sdf) or os.path.getsize(output_sdf) == 0:
        logging.warning(f"SMINA 출력 파일이 비어있음: {output_sdf}")
        return False

    return True


def smina_dock_individual(smiles_list: list, ids: list, protein_pdb: str,
                          output_dir: str, python_path: str,
                          center: tuple, box_size: float = 20.0,
                          exhaustiveness: int = 8) -> dict:
    """
    개별 SMILES를 각각 docking하여 결과 SDF 경로를 반환.
    일부 분자가 실패해도 나머지는 계속 진행.

    Args:
        smiles_list: SMILES 리스트
        ids: 분자 ID 리스트
        protein_pdb: 단백질 PDB 경로
        output_dir: docking 결과를 저장할 디렉토리
        python_path: RDKit이 설치된 Python 경로
        center: pocket 중심 좌표
        box_size: search box 크기
        exhaustiveness: SMINA 탐색 강도

    Returns:
        dict: {index: docked_sdf_path} (성공한 것만)
    """
    docked = {}

    for i, (smi, mol_id) in enumerate(zip(smiles_list, ids)):
        # 1) SMILES → SDF (RDKit으로 3D conformer 생성)
        input_sdf = os.path.join(output_dir, f"input_{i}.sdf")
        docked_sdf = os.path.join(output_dir, f"docked_{i}.sdf")

        convert_script = f"""
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

smi = "{smi}"
mol = Chem.MolFromSmiles(smi)
if mol is None:
    print("FAIL", file=sys.stderr)
    sys.exit(1)
mol = Chem.AddHs(mol)
res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
if res == -1:
    # ETKDGv3 실패 시 random coordinates로 재시도
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
mol.SetProp("_Name", "mol_{i}")
writer = Chem.SDWriter("{input_sdf}")
writer.write(mol)
writer.close()
print("OK")
"""
        r = subprocess.run(
            [python_path, "-c", convert_script],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or "FAIL" in (r.stderr or ""):
            log_smiles_error(smi, "RDKit 3D conformer 생성 실패", "SMINA")
            logging.warning(f"  [{i}] {mol_id}: SDF 변환 실패")
            continue

        # 2) SMINA docking
        success = smina_dock_sdf(
            input_sdf, protein_pdb, docked_sdf,
            center=center, box_size=box_size,
            exhaustiveness=exhaustiveness, num_modes=1,
        )

        if success:
            docked[i] = docked_sdf
            logging.info(f"  [{i}] {mol_id}: docking 성공")
        else:
            log_smiles_error(smi, "SMINA docking 실패", "SMINA")
            logging.warning(f"  [{i}] {mol_id}: docking 실패")

    logging.info(f"SMINA docking 완료: {len(docked)}/{len(smiles_list)} 성공")
    return docked


# ═══════════════════════════════════════════════════════════════
# Stub 모드 (모델 미설치 시 가짜 결과 생성)
# ═══════════════════════════════════════════════════════════════

# True이면 실제 모델 호출, False이면 stub (가짜) 결과 생성
# 각 모델을 설치한 뒤 해당 항목을 True로 변경하세요.
MODEL_INSTALLED = {
    "CheapNet_Cross":   True,
    "CheapNet_Diverse": True,
    "PIGNet2":          True,
    "OnionNet2":        True,
    "DeepDTA":          True,
    "GraphDTA":         True,
    "MGraphDTA":        True,
    "GIGN":             True,
    "PSICHIC":          True,
    "NMDN":             True,
}

import random
import hashlib


def generate_stub_score(smiles: str, model: str) -> float:
    """
    SMILES+모델명으로 결정론적인 가짜 스코어 생성.
    같은 입력이면 항상 같은 결과를 반환 (재현 가능).
    """
    seed = hashlib.md5(f"{smiles}:{model}".encode()).hexdigest()
    rng = random.Random(seed)
    # 대부분 affinity 모델은 pKd나 음의 binding energy를 출력
    # 현실적인 범위: -12.0 ~ -4.0 (pKd 스케일)
    return round(rng.uniform(-12.0, -4.0), 3)


def predict_stub(df: pd.DataFrame, smiles_col: str, model: str) -> pd.DataFrame:
    """Stub 모드: 가짜 affinity 스코어 생성"""
    logging.warning(f"⚠️  {model} STUB 모드 — 실제 모델이 설치되지 않았습니다")
    logging.warning(f"    설치 후 MODEL_INSTALLED['{model}'] = True로 변경하세요")

    results = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        score = generate_stub_score(smi, model)
        results.append({
            "ID": row.get("ID", ""),
            "SMILES": smi,
            "Affinity_Score": score,
            "Affinity_Model": f"{model}_STUB",
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 1 & 2: CheapNet (기존 코드 유지)
# ═══════════════════════════════════════════════════════════════

def predict_cheapnet(protein_file: str, ligands_file: str,
                     output_file: str, variant: str):
    """CheapNet Cross 또는 Diverse 실행"""
    variant_dir = "cross_dataset" if variant == "CheapNet_Cross" else "diverse_protein"
    cmd = [
        "python",
        f"external_models/CheapNet/{variant_dir}/predict.py",
        "--protein", protein_file,
        "--ligand", ligands_file,
        "--output", output_file,
    ]
    logging.info(f"CheapNet ({variant_dir}) 실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logging.info(f"stdout:\n{result.stdout[-500:]}")
    if result.stderr:
        logging.info(f"stderr:\n{result.stderr[-500:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"CheapNet ({variant_dir}) 실행 실패 (exit {result.returncode})"
        )


# ═══════════════════════════════════════════════════════════════
# 모델 3: PIGNet2 (Physics-informed GNN)
# ═══════════════════════════════════════════════════════════════
# 변경사항 (2026-04-03):
#   - SMINA docking 단계 추가 (RDKit conformer → SMINA docked pose → PIGNet2)
#   - README 권장: "We highly recommend to use SMINA-optimized ligand conformations"
#   - 개별 분자 docking 후 성공한 것만 PIGNet2에 전달

def predict_pignet2(df: pd.DataFrame, smiles_col: str,
                    protein_file: str) -> pd.DataFrame:
    """PIGNet2로 affinity 예측 (SMINA docking 포함)"""
    python_path = CONDA_PYTHON["PIGNet2"]
    pignet2_dir = os.path.abspath("external_models/PIGNet2")
    protein_abs = os.path.abspath(protein_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        dock_dir = os.path.join(tmpdir, "docking")
        os.makedirs(dock_dir)

        smiles_list = df[smiles_col].tolist()
        ids = df["ID"].tolist() if "ID" in df.columns else [f"mol_{i}" for i in range(len(df))]

        # ── 1) SMINA Docking ──
        logging.info("PIGNet2: SMINA docking 시작...")
        logging.info(f"  Pocket center: {POCKET_CENTER}")
        logging.info(f"  Box size: {POCKET_BOX_SIZE}Å")

        docked = smina_dock_individual(
            smiles_list=smiles_list,
            ids=ids,
            protein_pdb=protein_abs,
            output_dir=dock_dir,
            python_path=python_path,
            center=POCKET_CENTER,
            box_size=POCKET_BOX_SIZE,
            exhaustiveness=8,
        )

        if not docked:
            logging.error("PIGNet2: 모든 분자의 docking 실패!")
            return pd.DataFrame(columns=["ID", "SMILES", "Affinity_Score", "Affinity_Model"])

        # ── 2) Docked SDF 병합 ──
        # PIGNet2는 여러 SDF를 -l 옵션으로 전달할 수 있음
        # 또는 하나로 합쳐서 전달
        merged_sdf = os.path.join(tmpdir, "docked_all.sdf")
        idx_map = {}  # PIGNet2 출력의 mol 인덱스 → 원래 df 인덱스
        mol_counter = 0

        with open(merged_sdf, "w") as out_f:
            for orig_idx in sorted(docked.keys()):
                sdf_path = docked[orig_idx]
                with open(sdf_path) as in_f:
                    content = in_f.read()
                if content.strip():
                    out_f.write(content)
                    if not content.endswith("\n"):
                        out_f.write("\n")
                    idx_map[mol_counter] = orig_idx
                    mol_counter += 1

        logging.info(f"PIGNet2: {mol_counter}개 docked 분자 준비 완료")

        # ── 3) PIGNet2 예측 실행 ──
        out_path = os.path.join(tmpdir, "predict.txt")
        ckpt_path = os.path.join(pignet2_dir, "src", "ckpt", "pda_0.pt")
        cmd = [
            python_path,
            os.path.join(pignet2_dir, "src", "exe", "predict.py"),
            ckpt_path,
            "-p", protein_abs,
            "-l", merged_sdf,
            "-o", out_path,
            "--no-prot-pdb",
            "--no-prot-sdf",
        ]
        logging.info(f"PIGNet2 실행: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)

        if r.stdout:
            logging.info(f"PIGNet2 stdout:\n{r.stdout[-2000:]}")
        if r.stderr:
            logging.info(f"PIGNet2 stderr:\n{r.stderr[-2000:]}")

        if r.returncode != 0:
            raise RuntimeError(f"PIGNet2 실행 실패 (exit {r.returncode})")

        # ── 4) 결과 파싱 ──
        # PIGNet2 출력 형식: name  true(=0)  sum(pred)  vdw  hbond  metal  hydro  dummy
        # sum(pred)가 binding energy (kcal/mol) — 음수가 정상
        results = []
        with open(out_path) as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 3:
                    name = parts[0]
                    dG = float(parts[2])  # kcal/mol (합산된 binding energy)

                    # kcal/mol → pKd 변환: pKd = ΔG / -1.364 (at 298K)
                    pKd = dG / -1.364

                    # PIGNet2 출력 순서 → 원래 df 인덱스 매핑
                    # name format: protein_docked_all_{conf_idx}
                    # 하지만 merged SDF에서 순서대로 나오므로 line_num 사용
                    if line_num in idx_map:
                        orig_idx = idx_map[line_num]
                        results.append({
                            "ID": df.iloc[orig_idx].get("ID", f"mol_{orig_idx}"),
                            "SMILES": df.iloc[orig_idx][smiles_col],
                            "Affinity_Score": round(pKd, 3),
                            "Affinity_Model": "PIGNet2",
                        })
                        logging.info(
                            f"  {ids[orig_idx]}: dG={dG:.3f} kcal/mol → pKd={pKd:.3f}"
                        )
                    else:
                        logging.warning(
                            f"  PIGNet2 결과 line {line_num}: idx_map에 없음 (name={name})"
                        )

    logging.info(f"PIGNet2 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 4: OnionNet-2 (2D CNN)
# ═══════════════════════════════════════════════════════════════
# 변경사항 (2026-04-03):
#   - SMINA docking 단계 추가 (RDKit conformer → SMINA docked pose → OnionNet2)
#   - docked pose PDB를 OnionNet2에 전달

def predict_onionnet2(df: pd.DataFrame, smiles_col: str,
                      protein_file: str) -> pd.DataFrame:
    """OnionNet-2로 affinity 예측 (SMINA docking 포함)"""
    python_path = CONDA_PYTHON["OnionNet2"]
    onionnet_dir = os.path.abspath("external_models/OnionNet-2")
    protein_abs = os.path.abspath(protein_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        dock_dir = os.path.join(tmpdir, "docking")
        os.makedirs(dock_dir)

        smiles_list = df[smiles_col].tolist()
        ids = df["ID"].tolist() if "ID" in df.columns else [f"mol_{i}" for i in range(len(df))]

        # ── 1) SMINA Docking ──
        logging.info("OnionNet-2: SMINA docking 시작...")

        docked = smina_dock_individual(
            smiles_list=smiles_list,
            ids=ids,
            protein_pdb=protein_abs,
            output_dir=dock_dir,
            python_path=python_path,
            center=POCKET_CENTER,
            box_size=POCKET_BOX_SIZE,
            exhaustiveness=8,
        )

        if not docked:
            logging.error("OnionNet-2: 모든 분자의 docking 실패!")
            return pd.DataFrame(columns=["ID", "SMILES", "Affinity_Score", "Affinity_Model"])

        # ── 2) Docked SDF → PDB 변환 + OnionNet2 예측 ──
        predict_script = os.path.join(onionnet_dir, "scoring", "predict.py")
        model_path = os.path.join(onionnet_dir, "models", "62shell_saved-model.h5")
        scaler_path = os.path.join(onionnet_dir, "models", "train_scaler.scaler")

        results = []
        for orig_idx in sorted(docked.keys()):
            smi = smiles_list[orig_idx]
            mol_id = ids[orig_idx]
            docked_sdf = docked[orig_idx]

            # SDF → PDB 변환 (OnionNet2는 PDB 입력)
            lig_pdb = os.path.join(tmpdir, f"docked_{orig_idx}.pdb")
            convert_cmd = f"""
from rdkit import Chem
mol = Chem.MolFromMolFile("{docked_sdf}", removeHs=False)
if mol is not None:
    Chem.MolToPDBFile(mol, "{lig_pdb}")
    print("OK")
else:
    print("FAIL")
"""
            r = subprocess.run(
                [python_path, "-c", convert_cmd],
                capture_output=True, text=True,
            )
            if "FAIL" in (r.stdout or "") or not os.path.exists(lig_pdb):
                log_smiles_error(smi, "Docked SDF → PDB 변환 실패", "OnionNet2")
                continue

            # OnionNet2 예측
            out_csv = os.path.join(tmpdir, f"pred_{orig_idx}.csv")
            cmd = [
                python_path, predict_script,
                "-rec_fpath", protein_abs,
                "-lig_fpath", lig_pdb,
                "-model", model_path,
                "-scaler", scaler_path,
                "-shells", "62",
                "-shape", "84,124,1",
                "-out_fpath", out_csv,
            ]
            logging.info(f"OnionNet-2 예측 [{orig_idx+1}/{len(df)}]: {smi[:50]}")
            r = subprocess.run(cmd, capture_output=True, text=True)

            if r.returncode != 0:
                log_smiles_error(smi, f"예측 실패: {r.stderr[-200:]}", "OnionNet2")
                logging.warning(f"  OnionNet-2 실패: {r.stderr[-300:]}")
                continue

            # 결과 파싱
            try:
                import pandas as _pd
                pred_df = _pd.read_csv(out_csv)
                score = float(pred_df["pred_pKd"].iloc[0])
                results.append({
                    "ID": mol_id,
                    "SMILES": smi,
                    "Affinity_Score": score,
                    "Affinity_Model": "OnionNet2",
                })
            except Exception as e:
                log_smiles_error(smi, f"결과 파싱 실패: {e}", "OnionNet2")

    logging.info(f"OnionNet-2 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 5: DeepDTA (1D sequence-based CNN)
# ═══════════════════════════════════════════════════════════════
# 변경사항 (2026-04-03):
#   - 단백질 서열을 CDK2 → β2AR(4LDE)로 수정
#   - 서열은 target_protein.pdb Chain A에서 추출

# CDK2 서열 (Davis 데이터셋에서 직접 추출, proteins.txt key="CDK2")
# 길이: 298 aa
# 주의: UniProt P24941과 4곳 차이 (pos 123,137,139,140) — Davis 버전을 사용해야 함
TARGET_SEQUENCE = (
    "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNH"
    "PNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHS"
    "HRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYY"
    "STAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSF"
    "PKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
)


def predict_deepdta(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """DeepDTA로 affinity 예측 (별도 스크립트 파일 방식)"""
    python_path = CONDA_PYTHON["DeepDTA"]
    deepdta_dir = os.path.abspath("external_models/DeepDTA")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(tmpdir, "input.json")
        output_json = os.path.join(tmpdir, "output.json")
        pred_script_path = os.path.join(tmpdir, "predict_deepdta.py")

        # 입력 데이터 준비
        input_data = {
            "protein_sequence": TARGET_SEQUENCE,
            "smiles_list": df[smiles_col].tolist(),
        }
        with open(input_json, "w") as f:
            json.dump(input_data, f)

        # DeepDTA 예측 스크립트를 별도 .py 파일로 저장
        # (f-string 인라인 방식은 중괄호 이스케이핑 문제 발생)
        pred_script_content = r'''
import json, sys, os, pickle
import numpy as np

deepdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, os.path.join(deepdta_dir, "source"))

with open(input_json) as f:
    data = json.load(f)

protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

# 문자 인코딩
charset_dir = os.path.join(deepdta_dir, "data", "davis")

try:
    with open(os.path.join(charset_dir, "charsmiset.pkl"), "rb") as f:
        charsmiset = pickle.load(f)
    with open(os.path.join(charset_dir, "charseqset.pkl"), "rb") as f:
        charseqset = pickle.load(f)
    print("Loaded charset from pickle", file=sys.stderr)
except FileNotFoundError:
    SMILES_CHARS = " #%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVYZ[\\]abcdefghiklmnoprstuvyz"
    PROTEIN_CHARS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    charsmiset = {c: i for i, c in enumerate(SMILES_CHARS)}
    charseqset = {c: i for i, c in enumerate(PROTEIN_CHARS)}
    print("Using fallback charset", file=sys.stderr)

def encode_smiles(smi, charset, max_len=100):
    enc = [charset.get(c, 0) + 1 for c in smi[:max_len]]
    return enc + [0] * (max_len - len(enc))

def encode_protein(seq, charset, max_len=1000):
    enc = [charset.get(c, 0) + 1 for c in seq[:max_len]]
    return enc + [0] * (max_len - len(enc))

prot_enc = encode_protein(protein_seq, charseqset)
smi_encs = [encode_smiles(smi, charsmiset) for smi in smiles_list]

try:
    import tensorflow as tf
    def cindex_score(y_true, y_pred):
        import tensorflow.keras.backend as K
        g = y_true - K.expand_dims(y_true, -1)
        f = y_pred - K.expand_dims(y_pred, -1)
        g = K.cast(g > 0, K.floatx())
        f = K.cast(f > 0, K.floatx())
        return K.sum(g * f) / (K.sum(g) + K.epsilon())

    model_path = os.path.join(deepdta_dir, "source", "saved_models", "deepdta_fold0_p0_00.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={"cindex_score": cindex_score})

    X_drug = np.array(smi_encs)
    X_prot = np.tile(np.array(prot_enc), (len(smi_encs), 1))

    preds = model.predict([X_drug, X_prot]).flatten()
    results = [{"smiles": smi, "score": float(p)} for smi, p in zip(smiles_list, preds)]
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    results = []

with open(output_json, "w") as f:
    json.dump(results, f)
'''
        with open(pred_script_path, "w") as f:
            f.write(pred_script_content)

        logging.info("DeepDTA 예측 시작...")
        logging.info(f"  단백질 서열 길이: {len(TARGET_SEQUENCE)} aa (CDK2, P24941)")
        r = subprocess.run(
            [python_path, pred_script_path, deepdta_dir, input_json, output_json],
            capture_output=True, text=True,
        )

        if r.stderr:
            logging.info(f"DeepDTA stderr:\n{r.stderr[-2000:]}")

        if r.returncode != 0:
            raise RuntimeError(f"DeepDTA 실행 실패: {r.stderr[:2000]}")

        # 결과 파싱
        with open(output_json) as f:
            preds = json.load(f)

        results = []
        for i, pred in enumerate(preds):
            results.append({
                "ID": df.iloc[i].get("ID", f"mol_{i}"),
                "SMILES": pred["smiles"],
                "Affinity_Score": pred["score"],
                "Affinity_Model": "DeepDTA",
            })

    logging.info(f"DeepDTA 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 6: GraphDTA (GINConvNet — Drug graph GNN + Protein seq CNN)
# ═══════════════════════════════════════════════════════════════
# sequence-based, docking 불필요
# 입력: SMILES → 78차원 atom features 그래프 + 단백질 서열 정수 인코딩 (max_len=1000)
# 학습: Davis 데이터셋 (GINConvNet)
# conda: pignet2 환경 사용

def predict_graphdta(df, smiles_col):
    """GraphDTA (GINConvNet)로 affinity 예측"""
    python_path = CONDA_PYTHON["GraphDTA"]
    graphdta_dir = os.path.abspath("external_models/GraphDTA")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(tmpdir, "input.json")
        output_json = os.path.join(tmpdir, "output.json")
        pred_script_path = os.path.join(tmpdir, "predict_graphdta.py")

        # 입력 데이터 준비
        input_data = {
            "protein_sequence": TARGET_SEQUENCE,
            "smiles_list": df[smiles_col].tolist(),
        }
        with open(input_json, "w") as f:
            json.dump(input_data, f)

        pred_script_content = r'''
import json, sys, os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import networkx as nx
from rdkit import Chem

graphdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, graphdta_dir)
sys.path.insert(0, os.path.join(graphdta_dir, "models"))
from models.ginconv import GINConvNet

with open(input_json) as f:
    data = json.load(f)

protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al',
             'I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn',
             'H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) +
        [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    if len(edges) == 0:
        edge_index = torch.LongTensor([[0], [0]])
    else:
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        edge_index = torch.LongTensor(edge_index).transpose(1, 0)
    return c_size, torch.FloatTensor(features), edge_index

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GINConvNet().to(device)
model_path = os.path.join(graphdta_dir, "model_GINConvNet_davis.model")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_enc = seq_cat(protein_seq)
target_tensor = torch.LongTensor([target_enc])

results = []
for smi in smiles_list:
    try:
        graph_data = smile_to_graph(smi)
        if graph_data is None:
            results.append({"smiles": smi, "score": None, "error": "Invalid SMILES"})
            continue
        c_size, features, edge_index = graph_data
        pyg_data = Data(
            x=features, edge_index=edge_index,
            y=torch.FloatTensor([0]), target=target_tensor)
        pyg_data.__setitem__('c_size', torch.LongTensor([c_size]))
        loader = DataLoader([pyg_data], batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)
            score = float(pred.cpu().numpy().flatten()[0])
            results.append({"smiles": smi, "score": score})
    except Exception as e:
        results.append({"smiles": smi, "score": None, "error": str(e)})

with open(output_json, "w") as f:
    json.dump(results, f)
print(f"GraphDTA done: {len([r for r in results if r.get('score') is not None])}/{len(smiles_list)}", file=sys.stderr)
'''
        with open(pred_script_path, "w") as f:
            f.write(pred_script_content)

        logging.info("GraphDTA (GINConvNet) 예측 시작...")
        logging.info(f"  단백질 서열 길이: {len(TARGET_SEQUENCE)} aa")
        r = subprocess.run(
            [python_path, pred_script_path, graphdta_dir, input_json, output_json],
            capture_output=True, text=True,
        )
        if r.stderr:
            logging.info(f"GraphDTA stderr:\n{r.stderr[-2000:]}")
        if r.returncode != 0:
            raise RuntimeError(f"GraphDTA 실행 실패: {r.stderr[:2000]}")

        with open(output_json) as f:
            preds = json.load(f)

        results = []
        for i, pred in enumerate(preds):
            if pred.get("score") is not None:
                results.append({
                    "ID": df.iloc[i].get("ID", f"mol_{i}"),
                    "SMILES": pred["smiles"],
                    "Affinity_Score": pred["score"],
                    "Affinity_Model": "GraphDTA",
                })
            else:
                log_smiles_error(pred["smiles"],
                                 pred.get("error", "Unknown error"), "GraphDTA")

    logging.info(f"GraphDTA 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 7: MGraphDTA (Multiscale Graph Neural Network)
# ═══════════════════════════════════════════════════════════════
# sequence-based, docking 불필요
# 입력: SMILES → 22차원 node features + 6차원 edge features 그래프
#       + 단백질 서열 정수 인코딩 (max_len=1200)
# 학습: Davis 데이터셋
# conda: pignet2 환경 사용

def predict_mgraphdta(df, smiles_col):
    """MGraphDTA로 affinity 예측"""
    python_path = CONDA_PYTHON["MGraphDTA"]
    mgraphdta_dir = os.path.abspath("external_models/MGraphDTA/regression")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(tmpdir, "input.json")
        output_json = os.path.join(tmpdir, "output.json")
        pred_script_path = os.path.join(tmpdir, "predict_mgraphdta.py")

        input_data = {
            "protein_sequence": TARGET_SEQUENCE,
            "smiles_list": df[smiles_col].tolist(),
        }
        with open(input_json, "w") as f:
            json.dump(input_data, f)

        pred_script_content = r'''
import json, sys, os, glob
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

mgraphdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, mgraphdta_dir)
from model import MGraphDTA

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

with open(input_json) as f:
    data = json.load(f)

protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

VOCAB_PROTEIN = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,
                 "F":7,"I":8,"H":9,"K":10,"M":11,"L":12,
                 "O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,
                 "U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}

target_len = 1200
target_enc = [VOCAB_PROTEIN.get(s, 0) for s in protein_seq]
if len(target_enc) < target_len:
    target_enc = target_enc + [0] * (target_len - len(target_enc))
else:
    target_enc = target_enc[:target_len]
target_tensor = torch.LongTensor([target_enc])

def mol2graph(mol):
    if mol is None:
        return None
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(),
            acceptor=0, donor=0, aromatic=atom_i.GetIsAromatic(),
            hybridization=atom_i.GetHybridization(), num_h=atom_i.GetTotalNumHs(),
            ExplicitValence=atom_i.GetExplicitValence(), FormalCharge=atom_i.GetFormalCharge(),
            ImplicitValence=atom_i.GetImplicitValence(), NumExplicitHs=atom_i.GetNumExplicitHs(),
            NumRadicalElectrons=atom_i.GetNumRadicalElectrons())
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            for n in feats[i].GetAtomIds(): g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            for n in feats[i].GetAtomIds(): g.nodes[n]['acceptor'] = 1
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType(),
                           IsConjugated=int(e_ij.GetIsConjugated()))
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['H','C','N','O','F','Cl','S','Br','I']]
        h_t.append(d['a_num']); h_t.append(d['acceptor']); h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['hybridization'] == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['num_h']); h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge']); h_t.append(d['ImplicitValence'])
        h_t.append(d['NumExplicitHs']); h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    if node_attr.max() != node_attr.min():
        node_attr = (node_attr - node_attr.min()) / (node_attr.max() - node_attr.min())
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x) for x in (Chem.rdchem.BondType.SINGLE,
               Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
               Chem.rdchem.BondType.AROMATIC)]
        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t
    if len(e) == 0:
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.FloatTensor([[0,0,0,0,0,0]])
    else:
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
    return node_attr, edge_index, edge_attr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

# 모델 파일 찾기
model_candidates = glob.glob(os.path.join(mgraphdta_dir, "save", "**", "model", "*.pt"), recursive=True)
if not model_candidates:
    model_candidates = glob.glob(os.path.join(os.path.dirname(mgraphdta_dir),
                                               "visualization", "pretrained_model", "*.pt"))
if not model_candidates:
    print("ERROR: No MGraphDTA model file found!", file=sys.stderr)
    sys.exit(1)
model_path = sorted(model_candidates)[-1]
print(f"Loading MGraphDTA model: {model_path}", file=sys.stderr)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

results = []
for smi in smiles_list:
    try:
        mol = Chem.MolFromSmiles(smi)
        graph_data = mol2graph(mol)
        if graph_data is None:
            results.append({"smiles": smi, "score": None, "error": "Invalid SMILES"})
            continue
        node_attr, edge_index, edge_attr = graph_data
        pyg_data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.FloatTensor([0]), target=target_tensor)
        loader = DataLoader([pyg_data], batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)
            score = float(pred.cpu().numpy().flatten()[0])
            results.append({"smiles": smi, "score": score})
    except Exception as e:
        results.append({"smiles": smi, "score": None, "error": str(e)})

with open(output_json, "w") as f:
    json.dump(results, f)
print(f"MGraphDTA done: {len([r for r in results if r.get('score') is not None])}/{len(smiles_list)}", file=sys.stderr)
'''
        with open(pred_script_path, "w") as f:
            f.write(pred_script_content)

        logging.info("MGraphDTA 예측 시작...")
        logging.info(f"  단백질 서열 길이: {len(TARGET_SEQUENCE)} aa")
        r = subprocess.run(
            [python_path, pred_script_path, mgraphdta_dir, input_json, output_json],
            capture_output=True, text=True,
        )
        if r.stderr:
            logging.info(f"MGraphDTA stderr:\n{r.stderr[-2000:]}")
        if r.returncode != 0:
            raise RuntimeError(f"MGraphDTA 실행 실패: {r.stderr[:2000]}")

        with open(output_json) as f:
            preds = json.load(f)

        results = []
        for i, pred in enumerate(preds):
            if pred.get("score") is not None:
                results.append({
                    "ID": df.iloc[i].get("ID", f"mol_{i}"),
                    "SMILES": pred["smiles"],
                    "Affinity_Score": pred["score"],
                    "Affinity_Model": "MGraphDTA",
                })
            else:
                log_smiles_error(pred["smiles"],
                                 pred.get("error", "Unknown error"), "MGraphDTA")

    logging.info(f"MGraphDTA 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 8: GIGN (3D Interaction Graph Neural Network)
# ═══════════════════════════════════════════════════════════════
# structure-based, SMINA docking 필요
# 입력: docked pose (ligand PDB + protein pocket PDB) → 3D interaction graph
# 학습: PDBbind (pretrained)
# conda: pignet2 환경 사용

def predict_gign(df, smiles_col, protein_file):
    """GIGN으로 affinity 예측 (SMINA docking + pocket 추출 포함)"""
    python_path = CONDA_PYTHON["GIGN"]
    gign_dir = os.path.abspath("external_models/GIGN/GIGN")
    protein_abs = os.path.abspath(protein_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        dock_dir = os.path.join(tmpdir, "docking")
        os.makedirs(dock_dir)

        smiles_list = df[smiles_col].tolist()
        ids = df["ID"].tolist() if "ID" in df.columns else [f"mol_{i}" for i in range(len(df))]

        # ── 1) SMINA Docking ──
        logging.info("GIGN: SMINA docking 시작...")
        docked = smina_dock_individual(
            smiles_list=smiles_list,
            ids=ids,
            protein_pdb=protein_abs,
            output_dir=dock_dir,
            python_path=python_path,
            center=POCKET_CENTER,
            box_size=POCKET_BOX_SIZE,
            exhaustiveness=8,
        )

        if not docked:
            logging.error("GIGN: 모든 분자의 docking 실패!")
            return pd.DataFrame(columns=["ID", "SMILES", "Affinity_Score", "Affinity_Model"])

        # ── 2) 각 docked SDF에 대해 pocket 추출 + graph 생성 + 예측 ──
        input_json = os.path.join(tmpdir, "input.json")
        output_json = os.path.join(tmpdir, "output.json")
        pred_script_path = os.path.join(tmpdir, "predict_gign.py")

        # docked SDF 경로 매핑 준비
        docked_info = []
        for orig_idx in sorted(docked.keys()):
            docked_info.append({
                "orig_idx": orig_idx,
                "mol_id": ids[orig_idx],
                "smiles": smiles_list[orig_idx],
                "docked_sdf": docked[orig_idx],
            })

        input_data = {
            "protein_pdb": protein_abs,
            "pocket_distance": 5,
            "docked_info": docked_info,
            "gign_dir": gign_dir,
        }
        with open(input_json, "w") as f:
            json.dump(input_data, f)

        pred_script_content = r'''
import json, sys, os, pickle, subprocess
import numpy as np
import torch
import networkx as nx
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

input_json = sys.argv[1]
output_json = sys.argv[2]

with open(input_json) as f:
    data = json.load(f)

protein_pdb = data["protein_pdb"]
pocket_distance = data["pocket_distance"]
docked_info = data["docked_info"]
gign_dir = data["gign_dir"]

sys.path.insert(0, gign_dir)
from GIGN import GIGN
from utils import load_model_dict

# ── atom features (35 dim) ──
def one_of_k_encoding_unk(x, s):
    if x not in s: x = s[-1]
    return list(map(lambda a: x==a, s))

def atom_features_mol(mol, graph):
    atom_symbols = ['C','N','O','S','F','P','Cl','Br','I']
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols+['Unknown']) + \
                  one_of_k_encoding_unk(atom.GetDegree(),[0,1,2,3,4,5,6]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(),[0,1,2,3,4,5,6]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2]) + \
                  [atom.GetIsAromatic()] + \
                  one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])
        graph.add_node(atom.GetIdx(), feats=torch.FloatTensor(results))

def mol2graph(mol):
    graph = nx.Graph()
    atom_features_mol(mol, graph)
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    graph = graph.to_directed()
    x = torch.stack([f['feats'] for _,f in graph.nodes(data=True)])
    if graph.number_of_edges() > 0:
        ei = torch.stack([torch.LongTensor((u,v)) for u,v in graph.edges(data=False)]).T
    else:
        ei = torch.zeros((2,0), dtype=torch.long)
    return x, ei

def inter_graph(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dm = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dm < dis_threshold)
    graph = nx.Graph()
    for i,j in zip(node_idx[0], node_idx[1]):
        graph.add_edge(i, j+atom_num_l)
    graph = graph.to_directed()
    if graph.number_of_edges() > 0:
        ei = torch.stack([torch.LongTensor((u,v)) for u,v in graph.edges(data=False)]).T
    else:
        ei = torch.zeros((2,0), dtype=torch.long)
    return ei

from torch_geometric.data import Data, Batch

# ── 모델 로드 ──
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GIGN(35, 256).to(device)
import glob
model_candidates = glob.glob(os.path.join(gign_dir, "model", "**", "model", "*.pt"), recursive=True)
# best model = repeat1 (highest valid_pr)
best_model = [m for m in model_candidates if "repeat1" in m]
if best_model:
    model_path = best_model[0]
else:
    model_path = sorted(model_candidates)[-1]
print(f"Loading GIGN model: {model_path}", file=sys.stderr)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()

tmpdir = os.path.dirname(input_json)
results = []

for info in docked_info:
    orig_idx = info["orig_idx"]
    mol_id = info["mol_id"]
    smi = info["smiles"]
    docked_sdf = info["docked_sdf"]

    try:
        # SDF → ligand PDB
        lig_pdb = os.path.join(tmpdir, f"lig_{orig_idx}.pdb")
        subprocess.run(["obabel", docked_sdf, "-O", lig_pdb, "-d"],
                       capture_output=True, text=True, timeout=60)
        if not os.path.exists(lig_pdb):
            results.append({"orig_idx": orig_idx, "score": None, "error": "obabel fail"})
            continue

        # PyMOL로 pocket 추출
        pocket_pdb = os.path.join(tmpdir, f"pocket_{orig_idx}.pdb")
        pymol_script = f"""
import pymol
from pymol import cmd
cmd.load('{protein_pdb}', 'protein')
cmd.remove('resn HOH')
cmd.load('{lig_pdb}', 'ligand')
cmd.remove('hydrogens')
cmd.select('pocket', 'byres ligand around {pocket_distance}')
cmd.save('{pocket_pdb}', 'pocket')
cmd.delete('all')
"""
        pymol_script_path = os.path.join(tmpdir, f"pymol_{orig_idx}.py")
        with open(pymol_script_path, 'w') as f:
            f.write(pymol_script)
        r = subprocess.run([sys.executable, pymol_script_path],
                           capture_output=True, text=True, timeout=120)
        if not os.path.exists(pocket_pdb) or os.path.getsize(pocket_pdb) == 0:
            results.append({"orig_idx": orig_idx, "score": None, "error": "pocket extraction fail"})
            continue

        # RDKit로 로드
        ligand = Chem.MolFromPDBFile(lig_pdb, removeHs=True)
        pocket = Chem.MolFromPDBFile(pocket_pdb, removeHs=True)
        if ligand is None or pocket is None:
            results.append({"orig_idx": orig_idx, "score": None, "error": "RDKit load fail"})
            continue

        if ligand.GetNumConformers() == 0 or pocket.GetNumConformers() == 0:
            results.append({"orig_idx": orig_idx, "score": None, "error": "No conformer"})
            continue

        # Graph 생성
        atom_num_l = ligand.GetNumAtoms()
        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        x_l, ei_l = mol2graph(ligand)
        x_p, ei_p = mol2graph(pocket)

        x = torch.cat([x_l, x_p], dim=0)
        ei_intra = torch.cat([ei_l, ei_p + atom_num_l], dim=-1)
        ei_inter = inter_graph(ligand, pocket, dis_threshold=5.)
        pos = torch.cat([pos_l, pos_p], dim=0)

        if ei_inter.shape[1] == 0:
            results.append({"orig_idx": orig_idx, "score": None, "error": "No interaction edges"})
            continue

        pyg_data = Data(x=x, edge_index_intra=ei_intra, edge_index_inter=ei_inter,
                        y=torch.FloatTensor([0]), pos=pos)

        batch = Batch.from_data_list([pyg_data]).to(device)
        with torch.no_grad():
            pred = model(batch)
        score = float(pred.cpu().numpy().flatten()[0])
        results.append({"orig_idx": orig_idx, "score": score})

    except Exception as e:
        results.append({"orig_idx": orig_idx, "score": None, "error": str(e)})

with open(output_json, "w") as f:
    json.dump(results, f)
print(f"GIGN done: {len([r for r in results if r.get('score') is not None])}/{len(docked_info)}", file=sys.stderr)
'''
        with open(pred_script_path, "w") as f:
            f.write(pred_script_content)

        logging.info("GIGN 예측 시작...")
        r = subprocess.run(
            [python_path, pred_script_path, input_json, output_json],
            capture_output=True, text=True,
        )
        if r.stderr:
            logging.info(f"GIGN stderr:\n{r.stderr[-2000:]}")
        if r.returncode != 0:
            raise RuntimeError(f"GIGN 실행 실패: {r.stderr[:2000]}")

        with open(output_json) as f:
            preds = json.load(f)

        results = []
        for pred in preds:
            if pred.get("score") is not None:
                orig_idx = pred["orig_idx"]
                results.append({
                    "ID": ids[orig_idx],
                    "SMILES": smiles_list[orig_idx],
                    "Affinity_Score": pred["score"],
                    "Affinity_Model": "GIGN",
                })
            else:
                orig_idx = pred["orig_idx"]
                log_smiles_error(smiles_list[orig_idx],
                                 pred.get("error", "Unknown"), "GIGN")

    logging.info(f"GIGN 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 9: PSICHIC (PhySIcoCHemICal graph neural network)
# ═══════════════════════════════════════════════════════════════
# sequence-based, docking 불필요
# 입력: 단백질 서열 + SMILES → physicochemical graph
# 학습: PDBbind v2020 (pretrained)
# conda: psichic_fp 환경 사용

def predict_psichic(df, smiles_col):
    """PSICHIC으로 affinity 예측 (screening.py 활용)"""
    python_path = CONDA_PYTHON["PSICHIC"]
    psichic_dir = os.path.abspath("external_models/PSICHIC")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 입력 CSV 생성 (Protein, Ligand 컬럼)
        screen_csv = os.path.join(tmpdir, "screen.csv")
        screen_df = pd.DataFrame({
            "Protein": [TARGET_SEQUENCE] * len(df),
            "Ligand": df[smiles_col].tolist(),
        })
        screen_df.to_csv(screen_csv, index=False)

        result_dir = os.path.join(tmpdir, "result")

        logging.info("PSICHIC 예측 시작...")
        logging.info(f"  단백질 서열 길이: {len(TARGET_SEQUENCE)} aa")

        r = subprocess.run(
            [python_path, os.path.join(psichic_dir, "screening.py"),
             "--screenfile", screen_csv,
             "--trained_model_path", os.path.join(psichic_dir, "trained_weights", "PDBv2020_PSICHIC"),
             "--result_path", result_dir,
             "--device", "cuda:0"],
            capture_output=True, text=True,
            cwd=psichic_dir,
        )

        if r.stderr:
            logging.info(f"PSICHIC stderr:\n{r.stderr[-2000:]}")
        if r.returncode != 0:
            raise RuntimeError(f"PSICHIC 실행 실패: {r.stderr[:2000]}")

        # 결과 파싱
        result_csv = os.path.join(result_dir, "screening.csv")
        if not os.path.exists(result_csv):
            raise RuntimeError(f"PSICHIC 결과 파일 없음: {result_csv}")

        pred_df = pd.read_csv(result_csv)

        results = []
        for i, row in pred_df.iterrows():
            if i < len(df) and "predicted_binding_affinity" in row:
                score = float(row["predicted_binding_affinity"])
                results.append({
                    "ID": df.iloc[i].get("ID", f"mol_{i}"),
                    "SMILES": df.iloc[i][smiles_col],
                    "Affinity_Score": score,
                    "Affinity_Model": "PSICHIC",
                })

    logging.info(f"PSICHIC 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 모델 10: DiffDock-NMDN (Normalized Mixture Density Network)
# ═══════════════════════════════════════════════════════════════
# structure-based, SMINA docking 필요
# 입력: docked pose (SDF) + protein PDB → NMDN score + pKd
# 학습: PDBbind (pretrained)
# conda: nmdn 환경 사용

def predict_nmdn(df, smiles_col, protein_file):
    """DiffDock-NMDN으로 affinity 예측 (SMINA docking 포함)"""
    python_path = CONDA_PYTHON["NMDN"]
    nmdn_dir = os.path.abspath("external_models/DiffDock-NMDN")
    protein_abs = os.path.abspath(protein_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        dock_dir = os.path.join(tmpdir, "docking")
        os.makedirs(dock_dir)

        smiles_list = df[smiles_col].tolist()
        ids = df["ID"].tolist() if "ID" in df.columns else [f"mol_{i}" for i in range(len(df))]

        # ── 1) SMINA Docking ──
        logging.info("NMDN: SMINA docking 시작...")
        docked = smina_dock_individual(
            smiles_list=smiles_list,
            ids=ids,
            protein_pdb=protein_abs,
            output_dir=dock_dir,
            python_path=CONDA_PYTHON["PIGNet2"],
            center=POCKET_CENTER,
            box_size=POCKET_BOX_SIZE,
            exhaustiveness=8,
        )

        if not docked:
            logging.error("NMDN: 모든 분자의 docking 실패!")
            return pd.DataFrame(columns=["ID", "SMILES", "Affinity_Score", "Affinity_Model"])

        # ── 2) 각 docked SDF에 대해 NMDN 예측 ──
        results = []
        for orig_idx in sorted(docked.keys()):
            smi = smiles_list[orig_idx]
            mol_id = ids[orig_idx]
            docked_sdf = docked[orig_idx]

            output_csv = os.path.join(tmpdir, f"nmdn_pred_{orig_idx}.csv")
            cmd = [
                python_path, os.path.join(nmdn_dir, "predict.py"),
                "--prot", protein_abs,
                "--ligs", docked_sdf,
                "--save_csv", output_csv,
            ]
            logging.info(f"  NMDN 예측 [{orig_idx+1}/{len(df)}]: {mol_id}")
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=nmdn_dir)

            if r.returncode != 0:
                log_smiles_error(smi, f"NMDN 실행 실패: {r.stderr[-200:]}", "NMDN")
                logging.warning(f"  NMDN 실패: {r.stderr[-300:]}")
                continue

            try:
                pred_df = pd.read_csv(output_csv)
                score = float(pred_df["pKd-Score"].iloc[0])
                results.append({
                    "ID": mol_id,
                    "SMILES": smi,
                    "Affinity_Score": score,
                    "Affinity_Model": "NMDN",
                })
            except Exception as e:
                log_smiles_error(smi, f"결과 파싱 실패: {e}", "NMDN")

    logging.info(f"NMDN 완료: {len(results)}개 예측")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════

try:
    logging.info("=" * 60)
    logging.info(f"Affinity 예측 작업 시작 — 모델: {model_name}")
    logging.info(f"입력: {ligands_file}")
    logging.info(f"단백질: {protein_file}")
    logging.info(f"출력: {output_file}")
    logging.info("=" * 60)

    # ── 입력 CSV 로드 ──
    df = pd.read_csv(ligands_file)
    smiles_col = find_smiles_column(df)
    logging.info(f"입력 분자 수: {len(df)}, SMILES 컬럼: {smiles_col}")

    # ── Stub 모드 확인 ──
    is_installed = MODEL_INSTALLED.get(model_name, False)

    if not is_installed and model_name in MODEL_INSTALLED:
        # Stub 모드: 가짜 결과 생성
        df_result = predict_stub(df, smiles_col, model_name)

    # ── 모델별 분기 (실제 실행) ──
    elif model_name == "CheapNet_Cross":
        logging.info("CheapNet (Cross-dataset) 실행 중...")
        predict_cheapnet(protein_file, ligands_file, output_file, model_name)
        # CheapNet은 직접 output_file에 쓰므로 여기서 종료
        logging.info("Affinity 작업 정상 완료")
        import sys
        sys.exit(0)  # Snakemake는 exit(0)을 정상 종료로 인식

    elif model_name == "CheapNet_Diverse":
        logging.info("CheapNet (Diverse Protein) 실행 중...")
        predict_cheapnet(protein_file, ligands_file, output_file, model_name)
        logging.info("Affinity 작업 정상 완료")
        import sys
        sys.exit(0)

    elif model_name == "PIGNet2":
        df_result = predict_pignet2(df, smiles_col, protein_file)

    elif model_name == "OnionNet2":
        df_result = predict_onionnet2(df, smiles_col, protein_file)

    elif model_name == "DeepDTA":
        df_result = predict_deepdta(df, smiles_col)

    elif model_name == "GraphDTA":
        df_result = predict_graphdta(df, smiles_col)

    elif model_name == "MGraphDTA":
        df_result = predict_mgraphdta(df, smiles_col)

    elif model_name == "GIGN":
        df_result = predict_gign(df, smiles_col, protein_file)

    elif model_name == "PSICHIC":
        df_result = predict_psichic(df, smiles_col)

    elif model_name == "NMDN":
        df_result = predict_nmdn(df, smiles_col, protein_file)

    else:
        logging.warning(f"알 수 없는 모델: {model_name} → stub 출력")
        df_result = predict_stub(df, smiles_col, model_name)

    # ── 저장 ──
    df_result.to_csv(output_file, index=False)

    # ── 실행 요약 ──
    logging.info("=" * 60)
    logging.info("실행 요약:")
    logging.info(f"  모델:        {model_name}")
    logging.info(f"  설치 상태:   {'실제 실행' if is_installed else 'STUB 모드'}")
    logging.info(f"  입력 분자:   {len(df)}")
    logging.info(f"  예측 완료:   {len(df_result)}")
    if os.path.exists(error_log_path):
        with open(error_log_path) as f:
            error_count = sum(1 for _ in f)
        logging.info(f"  개별 에러:   {error_count}건 → {error_log_path}")
    logging.info(f"  결과 저장:   {output_file}")
    logging.info("=" * 60)
    logging.info("Affinity 작업 정상 완료")

except Exception as e:
    logging.error("=" * 60)
    logging.error("Affinity 작업 실패!")
    logging.error(f"모델: {model_name}")
    logging.error(f"에러: {e}")
    logging.error(traceback.format_exc())
    logging.error("=" * 60)
    raise e