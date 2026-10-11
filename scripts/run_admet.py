"""
run_admet.py — ADMET 예측 및 필터링 스크립트
==============================================
Snakemake rule 'admet'에서 호출됩니다.

각 모델은 별도 conda 환경에 설치되어 있으므로,
subprocess로 해당 환경의 Python을 직접 호출합니다.

베이스라인 모델:
  1. ADMET_AI  → conda env: ADMET_AI  (Chemprop-RDKit GNN, 41개 속성)
  2. Admetica  → conda env: Admetica  (Chemprop 기반, 25개 속성)

입력: results/01_gen/{gen}-molecules.csv  (컬럼: ID, SMILES, Gen_Model)
출력: results/02_admet/{gen}-{admet}-filtered.csv

로그 구조:
  logs/02_admet/{gen}-{admet}.log                  ← 전체 실행 로그
  logs/02_admet/errors/{gen}-{admet}-errors.jsonl   ← SMILES별 에러 상세
"""
import logging
import traceback
import subprocess
import tempfile
import os
import json
from datetime import datetime

import pandas as pd

# ─── Snakemake 인터페이스 ───
log_file    = snakemake.log[0]
input_file  = snakemake.input[0]
output_file = snakemake.output[0]
model_name  = snakemake.params.model_name

# ─── conda 환경별 Python 경로 ───
CONDA_PYTHON = {
    "ADMET_AI": "conda run -n ADMET_AI python",
    "Admetica": "conda run -n Admetica python",
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
# 필터링 기준
# ═══════════════════════════════════════════════════════════════
FILTERS = {
    # 컬럼명(소문자)     : (방향, 임계값)
    # "down" = 낮아야 통과, "up" = 높아야 통과
    "herg":              ("down", 0.5),
    "herg_karim":        ("down", 0.5),
    "ames":              ("down", 0.5),
    "dili":              ("down", 0.5),
    "bbb_martins":       ("up",   0.5),
    "bioavailability_ma":("up",   0.5),
    "hia_hou":           ("up",   0.5),
    "cyp2d6_veith":      ("down", 0.5),
    "cyp3a4_veith":      ("down", 0.5),
    "cyp2c9_veith":      ("down", 0.5),
    "cyp1a2-inhibitor":  ("down", 0.5),
    "cyp2c9-inhibitor":  ("down", 0.5),
    "cyp2c19-inhibitor": ("down", 0.5),
    "cyp2d6-inhibitor":  ("down", 0.5),
    "cyp3a4-inhibitor":  ("down", 0.5),
    "caco2_wang":        ("up",  -6.0),
    "caco2":             ("up",  -6.0),
}


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """필터링 — 컬럼명을 소문자로 매칭"""
    col_map = {col.lower(): col for col in df.columns}
    mask = pd.Series([True] * len(df), index=df.index)
    applied = {}

    for prop, (direction, threshold) in FILTERS.items():
        actual_col = col_map.get(prop)
        if actual_col is None:
            continue
        if direction == "down":
            mask &= df[actual_col] < threshold
            applied[prop] = f"< {threshold}"
        elif direction == "up":
            mask &= df[actual_col] >= threshold
            applied[prop] = f">= {threshold}"

    logging.info(f"적용된 필터 {len(applied)}개: {applied}")
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════
# 모델 1: ADMET-AI (subprocess로 별도 환경 호출)
# ═══════════════════════════════════════════════════════════════

def predict_admet_ai(input_csv: str, smiles_col: str) -> pd.DataFrame:
    """ADMET-AI 환경에서 예측 실행, 결과 CSV 반환"""
    python_path = CONDA_PYTHON["ADMET_AI"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_output = tmp.name

    # ADMET-AI 환경에서 실행할 Python 스크립트
    script = f"""
import pandas as pd
from admet_ai import ADMETModel
import json, sys

df = pd.read_csv("{input_csv}")
smiles_col = "{smiles_col}"
smiles_list = df[smiles_col].dropna().tolist()

model = ADMETModel()
results = []
errors = []

for smi in smiles_list:
    try:
        pred = model.predict(smiles=smi)
        pred["SMILES"] = smi
        results.append(pred)
    except Exception as e:
        errors.append({{"smiles": smi, "error": str(e)}})

if results:
    pd.DataFrame(results).to_csv("{tmp_output}", index=False)
else:
    pd.DataFrame().to_csv("{tmp_output}", index=False)

# 에러 정보를 stderr로 출력
if errors:
    print(json.dumps(errors), file=sys.stderr)
"""

    logging.info("ADMET-AI 예측 시작 (subprocess)...")
    result = subprocess.run(
        [python_path, "-c", script],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logging.error(f"ADMET-AI stdout: {result.stdout}")
        logging.error(f"ADMET-AI stderr: {result.stderr}")
        raise RuntimeError(f"ADMET-AI 실행 실패: {result.stderr[:500]}")

    # 개별 SMILES 에러 기록
    if result.stderr.strip():
        try:
            errors = json.loads(result.stderr.strip().split("\n")[-1])
            for err in errors:
                log_smiles_error(err["smiles"], err["error"], "ADMET_AI")
            logging.warning(f"ADMET-AI 개별 에러 {len(errors)}건")
        except json.JSONDecodeError:
            pass  # 일반 경고 메시지일 수 있음

    preds_df = pd.read_csv(tmp_output)
    os.unlink(tmp_output)
    logging.info(f"ADMET-AI 완료: {len(preds_df)}개 예측")
    return preds_df


# ═══════════════════════════════════════════════════════════════
# 모델 2: Admetica (subprocess로 별도 환경 호출)
# ═══════════════════════════════════════════════════════════════

# Admetica에서 사용 가능한 전체 속성 목록
ADMETICA_PROPERTIES = [
    "Caco2", "CL-Hepa", "CL-Micro", "Half-Life", "PPBR", "VDss",
    "hERG", "LD50",
    "Lipophilicity", "Solubility",
    "CYP1A2-Inhibitor", "CYP1A2-Substrate",
    "CYP2C9-Inhibitor", "CYP2C9-Substrate",
    "CYP2C19-Inhibitor", "CYP2C19-Substrate",
    "CYP2D6-Inhibitor", "CYP2D6-Substrate",
    "CYP3A4-Inhibitor", "CYP3A4-Substrate",
    "PGP-Inhibitor", "PGP-Substrate",
]


def predict_admetica(input_csv: str, smiles_col: str) -> pd.DataFrame:
    """Admetica CLI로 예측 실행"""
    python_path = CONDA_PYTHON["Admetica"]
    properties_str = ",".join(ADMETICA_PROPERTIES)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_output = tmp.name

    cmd = [
        python_path, "-m", "cli.cli",
        "--dataset-path", input_csv,
        "--smiles-column", smiles_col,
        "--properties", properties_str,
        "--save-path", tmp_output,
    ]

    logging.info(f"Admetica 예측 시작: {len(ADMETICA_PROPERTIES)}개 속성")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Admetica stderr: {result.stderr[:1000]}")

        # 실패한 속성을 하나씩 빼면서 재시도
        logging.info("속성별 개별 실행으로 전환...")
        input_df = pd.read_csv(input_csv)
        all_preds = {}

        for prop in ADMETICA_PROPERTIES:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp2:
                tmp2_path = tmp2.name

            cmd_single = [
                python_path, "-m", "cli.cli",
                "--dataset-path", input_csv,
                "--smiles-column", smiles_col,
                "--properties", prop,
                "--save-path", tmp2_path,
            ]
            r = subprocess.run(cmd_single, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(tmp2_path):
                try:
                    pred_df = pd.read_csv(tmp2_path)
                    for col in pred_df.columns:
                        all_preds[col] = pred_df[col]
                    logging.info(f"  {prop}: 성공")
                except Exception as e:
                    logging.warning(f"  {prop}: 결과 읽기 실패 — {e}")
            else:
                logging.warning(f"  {prop}: 실행 실패")

            if os.path.exists(tmp2_path):
                os.unlink(tmp2_path)

        if all_preds:
            preds_df = pd.DataFrame(all_preds)
        else:
            raise RuntimeError("Admetica: 모든 속성 예측 실패")

        logging.info(f"Admetica 완료: {len(preds_df.columns)}개 속성 예측")
        return preds_df

    preds_df = pd.read_csv(tmp_output)
    os.unlink(tmp_output)
    logging.info(f"Admetica 완료: {len(preds_df)}개 분자, {len(preds_df.columns)}개 속성")
    return preds_df


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════

try:
    logging.info("=" * 60)
    logging.info(f"ADMET 작업 시작 — 모델: {model_name}")
    logging.info(f"입력: {input_file}")
    logging.info(f"출력: {output_file}")
    logging.info("=" * 60)

    # ── 1) 입력 CSV 로드 ──
    df = pd.read_csv(input_file)
    smiles_col = find_smiles_column(df)
    logging.info(f"입력 분자 수: {len(df)}, SMILES 컬럼: {smiles_col}")

    # ── 2) 모델별 예측 ──
    if model_name == "ADMET_AI":
        preds_df = predict_admet_ai(input_file, smiles_col)

    elif model_name == "Admetica":
        preds_df = predict_admetica(input_file, smiles_col)

    else:
        logging.warning(f"알 수 없는 모델: {model_name} → dummy 출력")
        preds_df = df.copy()

    # ── 3) 원본 데이터와 병합 ──
    if "SMILES" in preds_df.columns:
        df_result = df.merge(preds_df, left_on=smiles_col, right_on="SMILES", how="inner")
    else:
        preds_df.index = range(len(preds_df))
        df_result = pd.concat(
            [df.reset_index(drop=True), preds_df.reset_index(drop=True)],
            axis=1,
        )

    logging.info(f"병합 후: {len(df_result)}개 분자, {len(df_result.columns)}개 컬럼")

    # ── 4) 필터링 ──
    df_filtered = apply_filters(df_result)
    df_filtered["ADMET_Pass"] = "Passed"
    df_filtered["ADMET_Model"] = model_name
    logging.info(f"필터링: {len(df_filtered)} / {len(df_result)} 통과")

    # ── 5) 저장 ──
    df_filtered.to_csv(output_file, index=False)

    # ── 6) 실행 요약 ──
    logging.info("=" * 60)
    logging.info("실행 요약:")
    logging.info(f"  모델:        {model_name}")
    logging.info(f"  입력 분자:   {len(df)}")
    logging.info(f"  예측 성공:   {len(df_result)}")
    logging.info(f"  필터 통과:   {len(df_filtered)}")
    logging.info(f"  통과율:      {len(df_filtered)/max(len(df_result),1)*100:.1f}%")
    if os.path.exists(error_log_path):
        with open(error_log_path) as f:
            error_count = sum(1 for _ in f)
        logging.info(f"  개별 에러:   {error_count}건 → {error_log_path}")
    logging.info(f"  결과 저장:   {output_file}")
    logging.info("=" * 60)
    logging.info("ADMET 작업 정상 완료")

except Exception as e:
    logging.error("=" * 60)
    logging.error("ADMET 작업 실패!")
    logging.error(f"모델: {model_name}")
    logging.error(f"에러: {e}")
    logging.error(traceback.format_exc())
    logging.error("=" * 60)
    raise e