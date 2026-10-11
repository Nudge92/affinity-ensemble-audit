"""
run_ensemble.py — Affinity 앙상블 스크립트
==============================================
Snakemake rule 'ensemble_affinity'에서 호출됩니다.

여러 Affinity 모델의 결과를 합쳐서 최종 후보를 선정합니다.

앙상블 전략:
  1. 각 모델별 순위(rank) 산출
  2. 평균 순위(mean rank) 기반 통합 랭킹
  3. 다수 모델에서 상위권인 분자 우선 선정
  4. Top-N 후보 출력

입력: results/03_affinity/{gen}-{admet}-{aff}-scores.csv (여러 개)
출력: results/04_ensemble/{gen}-{admet}-final_top_candidates.csv

로그: logs/04_ensemble/{gen}-{admet}-ensemble.log
"""
import logging
import traceback
import os

import pandas as pd
import numpy as np

# ─── Snakemake 인터페이스 ───
log_file    = snakemake.log[0]
input_files = snakemake.input.scores   # 리스트
output_file = snakemake.output[0]

# ─── 설정 ───
TOP_N = 50          # 최종 후보 수
MIN_MODELS = 2      # 최소 이 만큼의 모델에서 예측되어야 후보에 포함

# ═══════════════════════════════════════════════════════════════
# 로깅 설정
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Affinity 결과 파일 로드 및 검증"""
    df = pd.read_csv(filepath)

    # 필수 컬럼 확인
    required = {"SMILES", "Affinity_Score", "Affinity_Model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{filepath}: 필수 컬럼 누락 — {missing}. "
            f"현재 컬럼: {list(df.columns)}"
        )

    # Affinity_Score가 숫자인지 확인
    df["Affinity_Score"] = pd.to_numeric(df["Affinity_Score"], errors="coerce")
    n_invalid = df["Affinity_Score"].isna().sum()
    if n_invalid > 0:
        logging.warning(f"  {filepath}: {n_invalid}개 비수치 스코어 제거")
        df = df.dropna(subset=["Affinity_Score"])

    return df


def rank_within_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 내에서 순위 산출.
    Affinity_Score가 낮을수록 좋다고 가정 (binding energy 스케일).
    """
    df = df.copy()
    df["Rank"] = df["Affinity_Score"].rank(method="min", ascending=True)
    df["Percentile"] = df["Rank"] / len(df)
    return df


def ensemble_rank_aggregation(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    """
    여러 모델 결과를 통합하여 최종 랭킹 생성.

    전략:
    - 각 모델 내 백분위(percentile) 기준으로 정규화
    - SMILES별로 모든 모델의 percentile 평균 계산
    - 모델 수(coverage)도 고려: 더 많은 모델에서 예측된 분자에 보너스
    """
    ranked_dfs = []
    for df in all_results:
        model_name = df["Affinity_Model"].iloc[0]
        ranked = rank_within_model(df)
        ranked_dfs.append(ranked)

    # 전체 병합
    combined = pd.concat(ranked_dfs, ignore_index=True)

    # SMILES별 집계
    agg = combined.groupby("SMILES").agg(
        Mean_Percentile=("Percentile", "mean"),
        Mean_Score=("Affinity_Score", "mean"),
        Min_Score=("Affinity_Score", "min"),
        Max_Score=("Affinity_Score", "max"),
        Std_Score=("Affinity_Score", "std"),
        Num_Models=("Affinity_Model", "nunique"),
        Models=("Affinity_Model", lambda x: "|".join(sorted(x.unique()))),
    ).reset_index()

    # Std가 NaN인 경우 (단일 모델) 0으로 대체
    agg["Std_Score"] = agg["Std_Score"].fillna(0)

    # ID 복원 (첫 번째 발견된 ID 사용)
    id_map = combined.groupby("SMILES")["ID"].first().reset_index()
    agg = agg.merge(id_map, on="SMILES", how="left")

    # 최종 랭킹: Mean_Percentile 기준 (낮을수록 좋음)
    agg["Ensemble_Rank"] = agg["Mean_Percentile"].rank(method="min")
    agg = agg.sort_values("Ensemble_Rank")

    return agg


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════

try:
    logging.info("=" * 60)
    logging.info("Ensemble 작업 시작")
    logging.info(f"입력 파일 수: {len(input_files)}")
    for f in input_files:
        logging.info(f"  - {f}")
    logging.info(f"출력: {output_file}")
    logging.info("=" * 60)

    # ── 1) 모든 결과 로드 ──
    all_results = []
    for filepath in input_files:
        logging.info(f"로드 중: {filepath}")
        df = load_and_validate(filepath)
        model_name = df["Affinity_Model"].iloc[0] if len(df) > 0 else "unknown"
        logging.info(f"  모델: {model_name}, 분자 수: {len(df)}")
        if len(df) > 0:
            all_results.append(df)

    if not all_results:
        logging.error("유효한 결과 파일이 없습니다!")
        # 빈 파일 생성 (Snakemake 오류 방지)
        pd.DataFrame(columns=[
            "SMILES", "ID", "Ensemble_Rank", "Mean_Percentile",
            "Mean_Score", "Num_Models", "Models"
        ]).to_csv(output_file, index=False)
        raise RuntimeError("모든 입력 파일이 비어있습니다")

    # ── 2) 앙상블 랭킹 ──
    logging.info("앙상블 랭킹 계산 중...")
    ensemble_df = ensemble_rank_aggregation(all_results)

    # ── 3) 필터링: 최소 모델 수 ──
    before_filter = len(ensemble_df)
    ensemble_filtered = ensemble_df[
        ensemble_df["Num_Models"] >= min(MIN_MODELS, len(all_results))
    ].copy()
    logging.info(
        f"모델 수 필터 (>= {min(MIN_MODELS, len(all_results))}): "
        f"{len(ensemble_filtered)} / {before_filter}"
    )

    # 필터 후 분자가 없으면 전체 사용
    if len(ensemble_filtered) == 0:
        logging.warning("필터 후 분자가 없어 전체 결과를 사용합니다")
        ensemble_filtered = ensemble_df.copy()

    # ── 4) Top-N 선정 ──
    top_candidates = ensemble_filtered.head(TOP_N)

    # 출력 컬럼 정리
    output_cols = [
        "Ensemble_Rank", "ID", "SMILES",
        "Mean_Percentile", "Mean_Score", "Min_Score", "Max_Score", "Std_Score",
        "Num_Models", "Models",
    ]
    top_candidates = top_candidates[[c for c in output_cols if c in top_candidates.columns]]

    # ── 5) 저장 ──
    top_candidates.to_csv(output_file, index=False)

    # ── 6) 실행 요약 ──
    logging.info("=" * 60)
    logging.info("실행 요약:")
    logging.info(f"  Affinity 모델 수:   {len(all_results)}")
    logging.info(f"  전체 SMILES:        {len(ensemble_df)}")
    logging.info(f"  필터 통과:          {len(ensemble_filtered)}")
    logging.info(f"  최종 후보:          {len(top_candidates)}")
    if len(top_candidates) > 0:
        logging.info(f"  최고 후보 스코어:   {top_candidates['Mean_Score'].iloc[0]:.3f}")
        logging.info(f"  최고 후보 SMILES:   {top_candidates['SMILES'].iloc[0]}")
    logging.info(f"  결과 저장:          {output_file}")
    logging.info("=" * 60)
    logging.info("Ensemble 작업 정상 완료")

except Exception as e:
    logging.error("=" * 60)
    logging.error("Ensemble 작업 실패!")
    logging.error(f"에러: {e}")
    logging.error(traceback.format_exc())
    logging.error("=" * 60)
    raise e