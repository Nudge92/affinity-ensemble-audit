#!/usr/bin/env python3
"""
optimize_weights_v2.py — Phase 2: 타겟별 앙상블 가중치 최적화
================================================================
Phase 1에서 얻은 6타겟 validation CSV를 각각 읽어서:
  1) ρ < 0인 모델 제외
  2) 2모델/3모델/4모델 grid search → 타겟별 최적 조합
  3) 6타겟 결과를 종합 → 패밀리별 앙상블 레시피 도출

사용법:
  cd ~/Project/PLI
  python scripts/optimize_weights_v2.py

입력:
  results/06_validation/control_validation_{target}.csv  (6개)

출력:
  results/07_weighted_ensemble/phase2_optimization_report.txt
  results/07_weighted_ensemble/phase2_target_recipes.csv
"""
import os
import sys
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# ─── 설정 ───
PROJECT_DIR = os.path.expanduser("~/Project/PLI")
VALIDATION_DIR = os.path.join(PROJECT_DIR, "results", "06_validation")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "07_weighted_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ["cdk2", "egfr", "hiv1pr", "ptp1b", "era", "caii"]
TARGET_FAMILIES = {
    "cdk2": "kinase",
    "egfr": "kinase",
    "hiv1pr": "protease",
    "ptp1b": "phosphatase",
    "era": "nuclear_receptor",
    "caii": "metalloenzyme",
}

ALL_MODELS = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA", "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]

# Phase 1 가중치 최적화 규칙
RHO_EXCLUDE = 0.0    # ρ < 0 → 후보에서 제외
RHO_PASS = 0.3       # ρ ≥ 0.3 → 정상 최적화 대상

# grid search 설정
GRID_STEP = 0.05
MAX_MODELS_FULL_GRID = 4  # 4모델까지 full grid (5모델 이상은 조합 폭발)


# ═══════════════════════════════════════════════════════════════
# 핵심 함수
# ═══════════════════════════════════════════════════════════════

def weighted_score(row, weights, model_cols):
    """가중 평균 pKd 계산 (N/A 모델은 자동 스킵)"""
    vals, ws = [], []
    for col, w in zip(model_cols, weights):
        v = row[col]
        if not np.isnan(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if not ws:
        return np.nan
    return np.average(vals, weights=ws)


def evaluate_weights(df, weights, model_cols, target_col="Exp_pKd"):
    """특정 가중치 조합의 Spearman/Pearson/RMSE 평가"""
    preds = df.apply(lambda row: weighted_score(row, weights, model_cols), axis=1)
    valid = ~preds.isna() & ~df[target_col].isna()
    if valid.sum() < 3:
        return {"spearman": -999, "pearson": -999, "rmse": 999, "n": 0}

    exp = df.loc[valid, target_col].values
    pred = preds[valid].values
    sp, _ = spearmanr(exp, pred)
    pe, _ = pearsonr(exp, pred)
    rmse = np.sqrt(np.mean((exp - pred) ** 2))

    # 양성/음성 분리 gap
    df_tmp = df[valid].copy()
    df_tmp["_pred"] = pred
    pos = df_tmp[df_tmp["Type"] == "positive"]["_pred"]
    neg = df_tmp[df_tmp["Type"] == "negative"]["_pred"]
    gap = pos.mean() - neg.mean() if len(pos) > 0 and len(neg) > 0 else 0

    return {"spearman": sp, "pearson": pe, "rmse": rmse, "gap": gap, "n": int(valid.sum())}


def get_model_rho(df, target_col="Exp_pKd"):
    """각 모델의 개별 Spearman ρ 계산"""
    rhos = {}
    for m in ALL_MODELS:
        col = f"{m}_pKd"
        if col not in df.columns:
            continue
        pred = df[col].values
        exp = df[target_col].values
        valid = ~np.isnan(pred) & ~np.isnan(exp)
        if valid.sum() < 3:
            continue
        rho, _ = spearmanr(exp[valid], pred[valid])
        rhos[m] = rho
    return rhos


def grid_search_nmodel(df, model_cols, step=GRID_STEP):
    """N모델 grid search (가중치 합=1.0)"""
    n = len(model_cols)
    steps = int(1.0 / step) + 1
    weight_values = [round(i * step, 2) for i in range(steps)]

    best_sp = -999
    best_weights = None
    best_metrics = None
    count = 0

    for w_combo in itertools.product(weight_values, repeat=n):
        if abs(sum(w_combo) - 1.0) > 0.001:
            continue
        # 최소 가중치 필터: 0인 모델이 있으면 사실상 n-1모델 조합이므로 스킵
        # (n-1모델 조합은 별도로 탐색됨)
        if any(w == 0 for w in w_combo):
            continue
        count += 1
        metrics = evaluate_weights(df, w_combo, model_cols)
        if metrics["spearman"] > best_sp:
            best_sp = metrics["spearman"]
            best_weights = w_combo
            best_metrics = metrics

    return best_weights, best_metrics, count


def find_best_ensemble_for_target(df, rhos, max_n=MAX_MODELS_FULL_GRID):
    """
    타겟 하나에 대해 최적 앙상블 탐색.
    1) ρ < 0 모델 제외
    2) ρ ≥ 0 모델 중 2~max_n 모델 조합 grid search
    3) 전체 최고 조합 반환
    """
    # 후보 모델: ρ ≥ 0 만
    eligible = {m: r for m, r in rhos.items() if r >= RHO_EXCLUDE}
    # ρ 높은 순 정렬
    eligible_sorted = sorted(eligible.items(), key=lambda x: -x[1])

    if len(eligible_sorted) == 0:
        return None

    # 단독 모델 best
    best_solo = eligible_sorted[0]

    all_combos = []

    # 2모델 ~ max_n모델 조합 탐색
    for n in range(2, min(max_n + 1, len(eligible_sorted) + 1)):
        # 상위 모델로 조합 제한 (조합 폭발 방지)
        # n=2: 상위 6개에서 선택, n=3: 상위 5개, n=4: 상위 4개
        top_k = min(len(eligible_sorted), max(n + 2, 5))
        candidates = [m for m, _ in eligible_sorted[:top_k]]

        for combo in itertools.combinations(candidates, n):
            cols = [f"{m}_pKd" for m in combo]
            weights, metrics, count = grid_search_nmodel(df, cols)

            if weights is None:
                continue

            all_combos.append({
                "models": combo,
                "weights": weights,
                "metrics": metrics,
                "n_models": n,
                "n_searched": count,
            })

    if not all_combos:
        # 단독 모델만 반환
        return {
            "models": (best_solo[0],),
            "weights": (1.0,),
            "metrics": evaluate_weights(
                df, [1.0], [f"{best_solo[0]}_pKd"]
            ),
            "n_models": 1,
            "solo_rho": best_solo[1],
        }

    # Spearman 기준 정렬
    all_combos.sort(key=lambda x: -x["metrics"]["spearman"])

    best = all_combos[0]
    best["solo_rho"] = best_solo[1]
    best["all_combos"] = all_combos[:20]  # top-20 저장

    return best


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log("  Phase 2: 타겟별 앙상블 가중치 최적화")
    log("=" * 80)

    target_recipes = []  # CSV 출력용

    for target in TARGETS:
        csv_path = os.path.join(VALIDATION_DIR, f"control_validation_{target}.csv")
        if not os.path.exists(csv_path):
            log(f"\n⚠️ {target} CSV 없음: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        family = TARGET_FAMILIES[target]

        log(f"\n{'=' * 80}")
        log(f"  {target.upper()} ({family}) — n={len(df)}")
        log(f"{'=' * 80}")

        # ── 개별 모델 ρ 계산 ──
        rhos = get_model_rho(df)
        log(f"\n  개별 모델 Spearman ρ:")
        for m, r in sorted(rhos.items(), key=lambda x: -x[1]):
            status = "✅ pass" if r >= RHO_PASS else ("⚠️ weak" if r >= 0 else "❌ exclude")
            log(f"    {m:<12} ρ={r:+.3f}  {status}")

        excluded = [m for m, r in rhos.items() if r < RHO_EXCLUDE]
        if excluded:
            log(f"\n  제외 모델 (ρ<0): {', '.join(excluded)}")

        # ── 최적 앙상블 탐색 ──
        log(f"\n  Grid Search 진행 중...")
        best = find_best_ensemble_for_target(df, rhos)

        if best is None:
            log(f"  ⚠️ 사용 가능한 모델 없음!")
            continue

        # ── 결과 출력 ──
        models_str = " + ".join(best["models"])
        log(f"\n  ★ 최적 앙상블: {models_str}")
        log(f"    모델 수: {best['n_models']}")
        for m, w in zip(best["models"], best["weights"]):
            log(f"    {m:<12}: {w:.2f}")
        log(f"    Spearman:  {best['metrics']['spearman']:.3f}")
        log(f"    Pearson:   {best['metrics']['pearson']:.3f}")
        log(f"    RMSE:      {best['metrics']['rmse']:.3f}")
        log(f"    Gap:       {best['metrics']['gap']:+.3f}")

        # 개별 best 대비 향상
        best_solo_m = max(rhos.items(), key=lambda x: x[1])
        improvement = best["metrics"]["spearman"] - best_solo_m[1]
        log(f"\n    vs 최고 단독 모델 ({best_solo_m[0]}, ρ={best_solo_m[1]:.3f}):")
        log(f"    앙상블 향상: {improvement:+.3f}")

        # Top-5 조합 출력
        if "all_combos" in best and len(best["all_combos"]) > 1:
            log(f"\n  Top-5 조합:")
            log(f"    {'순위':>4}  {'조합':<40} {'Spearman':>10} {'Gap':>8}")
            log(f"    {'-'*4}  {'-'*40} {'-'*10} {'-'*8}")
            for i, combo in enumerate(best["all_combos"][:5]):
                combo_str = " + ".join(
                    [f"{m}({w:.2f})" for m, w in zip(combo["models"], combo["weights"])]
                )
                log(f"    {i+1:>4}  {combo_str:<40} "
                    f"{combo['metrics']['spearman']:>10.3f} "
                    f"{combo['metrics']['gap']:>+8.3f}")

        # ── 개별 모델 vs 앙상블 비교표 ──
        log(f"\n  전략 비교:")
        log(f"    {'전략':<35} {'Spearman':>10} {'RMSE':>8} {'Gap':>8}")
        log(f"    {'-'*35} {'-'*10} {'-'*8} {'-'*8}")

        # 개별 모델
        for m, r in sorted(rhos.items(), key=lambda x: -x[1]):
            m_metrics = evaluate_weights(df, [1.0], [f"{m}_pKd"])
            log(f"    {m + ' 단독':<35} {m_metrics['spearman']:>10.3f} "
                f"{m_metrics['rmse']:>8.3f} {m_metrics['gap']:>+8.3f}")

        # 동일가중 (eligible 모델만)
        eligible_models = [m for m, r in rhos.items() if r >= RHO_EXCLUDE]
        if len(eligible_models) >= 2:
            eq_cols = [f"{m}_pKd" for m in eligible_models]
            eq_w = [1.0 / len(eligible_models)] * len(eligible_models)
            eq_metrics = evaluate_weights(df, eq_w, eq_cols)
            eq_name = f"동일가중 {len(eligible_models)}모델"
            log(f"    {eq_name:<35} {eq_metrics['spearman']:>10.3f} "
                f"{eq_metrics['rmse']:>8.3f} {eq_metrics['gap']:>+8.3f}")

        # 최적 앙상블
        opt_name = f"★ 최적 {models_str}"
        log(f"    {opt_name:<35} {best['metrics']['spearman']:>10.3f} "
            f"{best['metrics']['rmse']:>8.3f} {best['metrics']['gap']:>+8.3f}")

        # recipe 저장
        recipe = {
            "target": target,
            "family": family,
            "n_models": best["n_models"],
            "models": " + ".join(best["models"]),
            "weights": " / ".join([f"{w:.2f}" for w in best["weights"]]),
            "ensemble_rho": round(best["metrics"]["spearman"], 3),
            "ensemble_rmse": round(best["metrics"]["rmse"], 3),
            "ensemble_gap": round(best["metrics"]["gap"], 3),
            "best_solo_model": best_solo_m[0],
            "best_solo_rho": round(best_solo_m[1], 3),
            "improvement": round(improvement, 3),
        }
        # 개별 가중치 컬럼
        for m, w in zip(best["models"], best["weights"]):
            recipe[f"w_{m}"] = w
        target_recipes.append(recipe)

    # ═══════════════════════════════════════════════════════════════
    # 종합 요약
    # ═══════════════════════════════════════════════════════════════
    log(f"\n\n{'#' * 80}")
    log(f"  Phase 2 종합 요약")
    log(f"{'#' * 80}")

    # 타겟별 레시피 요약표
    log(f"\n  {'타겟':<8} {'패밀리':<18} {'최적 조합':<35} {'ρ_ens':>8} {'ρ_solo':>8} {'향상':>8}")
    log(f"  {'-'*8} {'-'*18} {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for r in target_recipes:
        log(f"  {r['target']:<8} {r['family']:<18} {r['models']:<35} "
            f"{r['ensemble_rho']:>8.3f} {r['best_solo_rho']:>8.3f} {r['improvement']:>+8.3f}")

    # 패밀리별 공통 패턴 분석
    log(f"\n  패밀리별 모델 출현 빈도:")
    family_models = defaultdict(lambda: defaultdict(int))
    for r in target_recipes:
        for m in r["models"].split(" + "):
            family_models[r["family"]][m] += 1
            family_models["전체"][m] += 1

    for fam in sorted(family_models.keys()):
        models_sorted = sorted(family_models[fam].items(), key=lambda x: -x[1])
        models_str = ", ".join([f"{m}({c})" for m, c in models_sorted])
        log(f"    {fam:<18}: {models_str}")

    # 추천 레시피
    log(f"\n  추천 앙상블 레시피 (Phase 3 입력):")
    families_seen = {}
    for r in target_recipes:
        fam = r["family"]
        if fam not in families_seen:
            families_seen[fam] = r
        else:
            # 같은 패밀리면 ρ 높은 쪽 우선
            if r["ensemble_rho"] > families_seen[fam]["ensemble_rho"]:
                families_seen[fam] = r

    for fam, r in sorted(families_seen.items()):
        w_str = ", ".join([f"{m}={w}" for m, w in
                          zip(r["models"].split(" + "), r["weights"].split(" / "))])
        log(f"    {fam:<18}: {w_str}  (ρ={r['ensemble_rho']:.3f}, from {r['target']})")

    # Universal fallback
    # 전체에서 가장 많이 등장한 모델 조합
    model_freq = family_models["전체"]
    top_models = sorted(model_freq.items(), key=lambda x: -x[1])
    log(f"\n    Universal fallback 후보: {', '.join([f'{m}({c}/6)' for m, c in top_models[:3]])}")

    # ── 보고서 저장 ──
    report_path = os.path.join(OUTPUT_DIR, "phase2_optimization_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n보고서 저장: {report_path}")

    # ── 레시피 CSV 저장 ──
    recipe_df = pd.DataFrame(target_recipes)
    recipe_path = os.path.join(OUTPUT_DIR, "phase2_target_recipes.csv")
    recipe_df.to_csv(recipe_path, index=False)
    print(f"레시피 CSV 저장: {recipe_path}")

    print("\n" + "=" * 80)
    print("  Phase 2 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()