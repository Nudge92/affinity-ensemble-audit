#!/usr/bin/env python3
"""
run_ensemble_weighted.py — Phase 4: 가중 앙상블 스크립트
=========================================================
Snakefile의 ensemble_affinity 룰에서 호출됩니다.
Phase 3 레시피의 가중치를 적용하여 최종 후보를 순위화합니다.

입력: 여러 affinity score CSV (Snakemake input.scores)
출력: 가중 앙상블 순위 CSV

Snakemake params:
  weights_json: '{"PSICHIC": 0.70, "NMDN": 0.05, ...}'
  target_name: 'cdk2'
  family: 'kinase'
  excluded: '["DeepDTA"]'
"""
import os
import sys
import json

import pandas as pd
import numpy as np

# Snakemake에서 호출 시
try:
    score_files = snakemake.input.scores
    output_path = snakemake.output[0]
    log_path = snakemake.log[0]
    weights = json.loads(snakemake.params.weights_json)
    target_name = snakemake.params.target_name
    family = snakemake.params.family
    excluded = json.loads(snakemake.params.excluded)
except NameError:
    # 단독 실행 테스트용
    print("이 스크립트는 Snakemake에서 호출됩니다.")
    sys.exit(1)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

log_lines = []
def log(msg):
    log_lines.append(msg)
    print(msg)


log(f"=" * 70)
log(f"  가중 앙상블: {target_name} ({family})")
log(f"=" * 70)
log(f"  레시피: {weights}")
log(f"  제외: {excluded}")

# ── 1. Score 파일 로드 ──
model_dfs = {}
for fpath in score_files:
    # 파일명에서 모델명 추출: {gen}-{admet}-{aff}-scores.csv
    basename = os.path.basename(fpath)
    parts = basename.replace("-scores.csv", "").split("-")
    if len(parts) >= 3:
        model_name = parts[-1]
    else:
        model_name = basename

    if model_name in excluded:
        log(f"  {model_name}: 제외됨 (ρ<0)")
        continue

    try:
        df = pd.read_csv(fpath)
        if len(df) > 0:
            model_dfs[model_name] = df
            log(f"  {model_name}: {len(df)}개 분자 로드")
        else:
            log(f"  {model_name}: 빈 파일")
    except Exception as e:
        log(f"  {model_name}: 로드 실패 ({e})")

if not model_dfs:
    log("  ⚠️ 로드된 모델 없음!")
    pd.DataFrame().to_csv(output_path, index=False)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    sys.exit(0)

# ── 2. SMILES 기준 병합 ──
all_smiles = set()
for mdf in model_dfs.values():
    if "SMILES" in mdf.columns:
        all_smiles.update(mdf["SMILES"].dropna().tolist())

log(f"\n  총 고유 분자: {len(all_smiles)}개")

rows = []
for smi in all_smiles:
    row = {"SMILES": smi}

    # 각 모델 score 수집
    weighted_scores = []
    weighted_ws = []

    for model_name, mdf in model_dfs.items():
        match = mdf[mdf["SMILES"] == smi]
        if len(match) == 0:
            continue

        # Score 컬럼 탐색 (모델마다 컬럼명이 다를 수 있음)
        score_col = None
        for candidate in ["Affinity_Score", "pKd", "score", "predicted_pKd",
                          f"{model_name}_pKd", f"{model_name}_Score"]:
            if candidate in match.columns:
                score_col = candidate
                break

        if score_col is None:
            # 숫자 컬럼 중 마지막 것 사용
            num_cols = match.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                score_col = num_cols[-1]

        if score_col is None:
            continue

        score = float(match.iloc[0][score_col])
        row[f"{model_name}_Score"] = score

        # ID 가져오기
        if "ID" in match.columns:
            row["ID"] = match.iloc[0]["ID"]

        # 가중치 적용
        w = weights.get(model_name, 0)
        if w > 0 and not np.isnan(score):
            weighted_scores.append(score)
            weighted_ws.append(w)

    if weighted_scores:
        row["Weighted_Score"] = round(np.average(weighted_scores, weights=weighted_ws), 3)
    else:
        row["Weighted_Score"] = np.nan

    rows.append(row)

# ── 3. 순위화 ──
result_df = pd.DataFrame(rows)
result_df = result_df.dropna(subset=["Weighted_Score"])

if len(result_df) == 0:
    log("  ⚠️ 유효한 점수 없음!")
    result_df.to_csv(output_path, index=False)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    sys.exit(0)

# 높은 pKd가 좋으므로 내림차순
result_df["Weighted_Rank"] = result_df["Weighted_Score"].rank(ascending=False, method="min").astype(int)
result_df = result_df.sort_values("Weighted_Rank")

# 메타데이터 추가
result_df["Target"] = target_name
result_df["Family"] = family
result_df["Ensemble_Recipe"] = " + ".join(
    [f"{m}({w})" for m, w in weights.items()]
)

# ── 4. 출력 ──
result_df.to_csv(output_path, index=False)

log(f"\n  결과: {len(result_df)}개 후보")
log(f"  Top-5:")
for _, r in result_df.head(5).iterrows():
    model_scores = " | ".join(
        [f"{m}={r.get(f'{m}_Score', 'N/A')}" for m in weights.keys()
         if f"{m}_Score" in r.index and not pd.isna(r.get(f"{m}_Score"))]
    )
    log(f"    Rank {int(r['Weighted_Rank']):>3}: "
        f"Score={r['Weighted_Score']:.3f}  [{model_scores}]")

log(f"\n  저장: {output_path}")
log(f"=" * 70)

with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))