"""
run_gen.py — 분자 생성 스크립트
==============================================
Snakemake rule 'generate'에서 호출됩니다.

베이스라인 모델:
  1. Pocket2Mol → conda env: Pocket2Mol (E(3)-equivariant GNN)

입력: data/target_protein.pdb + 포켓 좌표
출력: results/01_gen/{gen}-molecules.csv  (컬럼: ID, SMILES, Gen_Model)

로그 구조:
  logs/01_gen/{gen}.log                  ← 전체 실행 로그
  logs/01_gen/errors/{gen}-errors.jsonl   ← 에러 상세
"""
import logging
import traceback
import subprocess
import os
import json
import glob
from datetime import datetime

import pandas as pd

# ─── Snakemake 인터페이스 ───
log_file    = snakemake.log[0]
output_file = snakemake.output[0]
model_name  = snakemake.params.model_name

# Pocket2Mol 전용 파라미터 (Snakefile에서 전달)
pdb_path    = snakemake.params.get("pdb_path", "data/target_protein.pdb")
pocket_center = snakemake.params.get("pocket_center", "0.0,0.0,0.0")

# ─── conda 환경별 Python 경로 ───
CONDA_PYTHON = {
    "Pocket2Mol": "conda run -n Pocket2Mol python",
}

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


# ═══════════════════════════════════════════════════════════════
# 모델 1: Pocket2Mol
# ═══════════════════════════════════════════════════════════════

def generate_pocket2mol(pdb_path: str, center: str) -> pd.DataFrame:
    """
    Pocket2Mol로 분자 생성
    - sample_for_pdb.py를 subprocess로 호출
    - outputs/ 폴더에서 SMILES.txt를 읽어 DataFrame 반환
    """
    python_path = CONDA_PYTHON["Pocket2Mol"]
    pocket2mol_dir = os.path.abspath("external_models/Pocket2Mol")
    pdb_abs_path = os.path.abspath(pdb_path)

    # 실행 전 기존 outputs 폴더 목록 기록 (새로 생성된 폴더를 찾기 위해)
    output_dir = os.path.join(pocket2mol_dir, "outputs")
    existing_dirs = set()
    if os.path.exists(output_dir):
        existing_dirs = set(os.listdir(output_dir))

    # Pocket2Mol 실행
    cmd = [
        "taskset", "-c", "0",
        python_path, "sample_for_pdb.py",
        "--pdb_path", pdb_abs_path,
        "--center", f" {center}",
    ]

    logging.info(f"Pocket2Mol 실행: center={center}")
    logging.info(f"명령어: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=pocket2mol_dir,
    )

    # 로그 기록
    if result.stdout:
        logging.info(f"Pocket2Mol stdout:\n{result.stdout[-500:]}")
    if result.stderr:
        logging.info(f"Pocket2Mol stderr:\n{result.stderr[-500:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Pocket2Mol 실행 실패 (exit code {result.returncode})")

    # 새로 생성된 출력 폴더 찾기
    current_dirs = set(os.listdir(output_dir))
    new_dirs = current_dirs - existing_dirs

    if not new_dirs:
        raise RuntimeError("Pocket2Mol 출력 폴더를 찾을 수 없습니다")

    # 가장 최근 폴더 선택
    new_dir = sorted(new_dirs)[-1]
    smiles_path = os.path.join(output_dir, new_dir, "SMILES.txt")

    if not os.path.exists(smiles_path):
        raise RuntimeError(f"SMILES.txt를 찾을 수 없습니다: {smiles_path}")

    # SMILES.txt 읽기
    with open(smiles_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    logging.info(f"Pocket2Mol 생성 완료: {len(smiles_list)}개 분자")

    # DataFrame 생성
    df = pd.DataFrame({
        "ID": [f"P2M_{i+1:04d}" for i in range(len(smiles_list))],
        "SMILES": smiles_list,
        "Gen_Model": "Pocket2Mol",
    })

    return df


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════

try:
    logging.info("=" * 60)
    logging.info(f"Generation 작업 시작 — 모델: {model_name}")
    logging.info(f"출력: {output_file}")
    logging.info("=" * 60)

    # ── 모델별 분기 ──
    if model_name == "Pocket2Mol":
        df = generate_pocket2mol(pdb_path, pocket_center)

    # ── 향후 추가 모델 ──
    # elif model_name == "TargetDiff":
    #     df = generate_targetdiff(pdb_path, pocket_center)

    else:
        # dummy 출력 (기존 동작 유지)
        logging.warning(f"알 수 없는 모델: {model_name} → dummy 출력")
        df = pd.DataFrame({
            "ID": [1],
            "SMILES": ["CCO"],
            "Gen_Model": [model_name],
        })

    # ── 저장 ──
    df.to_csv(output_file, index=False)

    # ── 실행 요약 ──
    logging.info("=" * 60)
    logging.info("실행 요약:")
    logging.info(f"  모델:          {model_name}")
    logging.info(f"  생성 분자 수:  {len(df)}")
    logging.info(f"  고유 SMILES:   {df['SMILES'].nunique()}")
    logging.info(f"  결과 저장:     {output_file}")
    logging.info("=" * 60)
    logging.info("Generation 작업 정상 완료")

except Exception as e:
    logging.error("=" * 60)
    logging.error("Generation 작업 실패!")
    logging.error(f"모델: {model_name}")
    logging.error(f"에러: {e}")
    logging.error(traceback.format_exc())
    logging.error("=" * 60)
    raise e