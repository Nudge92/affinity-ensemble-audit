#!/usr/bin/env python3
"""
cross_validate_recipes.py — Phase 2.5: 앙상블 레시피 교차 검증
================================================================
Phase 2에서 도출된 타겟별 최적 레시피를 다른 타겟에 교차 적용하여
일반화 성능을 평가합니다.

산출물:
  1) 6×6 교차 매트릭스 (레시피 × 타겟)
  2) Universal 레시피 후보 평가
  3) 패밀리 내 레시피 호환성 분석
  4) 최종 추천 레시피

사용법:
  cd ~/Project/PLI
  python scripts/cross_validate_recipes.py

입력:
  results/06_validation/control_validation_{target}.csv
  results/07_weighted_ensemble/phase2_target_recipes.csv

출력:
  results/07_weighted_ensemble/phase2.5_cross_validation_report.txt
  results/07_weighted_ensemble/phase2.5_cross_matrix.csv
"""
import os
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ─── 설정 ───
PROJECT_DIR = os.path.expanduser("~/Project/PLI")
VALIDATION_DIR = os.path.join(PROJECT_DIR, "results", "06_validation")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "07_weighted_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ["cdk2", "egfr", "hiv1pr", "ptp1b", "era", "caii"]
TARGET_LABELS = {
    "cdk2": "CDK2 (kinase)",
    "egfr": "EGFR (kinase)",
    "hiv1pr": "HIV1PR (protease)",
    "ptp1b": "PTP1B (phosphatase)",
    "era": "ERα (nuclear_rec)",
    "caii": "CA-II (metalloenz)",
}
TARGET_FAMILIES = {
    "cdk2": "kinase",
    "egfr": "kinase",
    "hiv1pr": "protease",
    "ptp1b": "phosphatase",
    "era": "nuclear_receptor",
    "caii": "metalloenzyme",
}

# Phase 2 최적 레시피 (결과에서 직접 가져옴)
PHASE2_RECIPES = {
    "cdk2": {
        "models": ["PSICHIC", "NMDN", "GIGN", "PIGNet2"],
        "weights": [0.70, 0.05, 0.10, 0.15],
    },
    "egfr": {
        "models": ["GraphDTA", "OnionNet2", "PSICHIC", "PIGNet2"],
        "weights": [0.40, 0.40, 0.10, 0.10],
    },
    "hiv1pr": {
        "models": ["OnionNet2", "NMDN", "GIGN", "PSICHIC"],
        "weights": [0.45, 0.15, 0.15, 0.25],
    },
    "ptp1b": {
        "models": ["MGraphDTA", "DeepDTA", "OnionNet2", "GraphDTA"],
        "weights": [0.40, 0.35, 0.05, 0.20],
    },
    "era": {
        "models": ["NMDN", "GIGN", "MGraphDTA", "PIGNet2"],
        "weights": [0.05, 0.10, 0.80, 0.05],
    },
    "caii": {
        "models": ["PSICHIC", "GIGN", "MGraphDTA", "GraphDTA"],
        "weights": [0.30, 0.05, 0.30, 0.35],
    },
}

# Universal 후보 레시피들 (Phase 2 출현빈도 + 직관적 조합)
UNIVERSAL_CANDIDATES = {
    "Uni-A: PSICHIC+GIGN (equal)": {
        "models": ["PSICHIC", "GIGN"],
        "weights": [0.50, 0.50],
    },
    "Uni-B: PSICHIC+GIGN+MGraphDTA (equal)": {
        "models": ["PSICHIC", "GIGN", "MGraphDTA"],
        "weights": [0.33, 0.33, 0.34],
    },
    "Uni-C: PSICHIC+GIGN+NMDN (equal)": {
        "models": ["PSICHIC", "GIGN", "NMDN"],
        "weights": [0.33, 0.33, 0.34],
    },
    "Uni-D: PSICHIC+GIGN+MGraphDTA+NMDN": {
        "models": ["PSICHIC", "GIGN", "MGraphDTA", "NMDN"],
        "weights": [0.30, 0.25, 0.25, 0.20],
    },
    "Uni-E: Top5 equal": {
        "models": ["PSICHIC", "GIGN", "MGraphDTA", "NMDN", "OnionNet2"],
        "weights": [0.20, 0.20, 0.20, 0.20, 0.20],
    },
    "Uni-F: All8 equal": {
        "models": ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA",
                    "MGraphDTA", "GIGN", "PSICHIC", "NMDN"],
        "weights": [0.125] * 8,
    },
}


def weighted_score(row, models, weights):
    """가중 평균 pKd 계산"""
    vals, ws = [], []
    for m, w in zip(models, weights):
        col = f"{m}_pKd"
        if col in row.index:
            v = row[col]
            if not np.isnan(v) and w > 0:
                vals.append(v)
                ws.append(w)
    if not ws:
        return np.nan
    return np.average(vals, weights=ws)


def evaluate_recipe(df, models, weights):
    """레시피의 Spearman ρ, gap 평가"""
    preds = df.apply(lambda row: weighted_score(row, models, weights), axis=1)
    valid = ~preds.isna() & ~df["Exp_pKd"].isna()
    n = valid.sum()
    if n < 3:
        return {"rho": np.nan, "gap": np.nan, "n": n}

    exp = df.loc[valid, "Exp_pKd"].values
    pred = preds[valid].values
    rho, _ = spearmanr(exp, pred)

    df_tmp = df[valid].copy()
    df_tmp["_pred"] = pred
    pos = df_tmp[df_tmp["Type"] == "positive"]["_pred"]
    neg = df_tmp[df_tmp["Type"] == "negative"]["_pred"]
    gap = pos.mean() - neg.mean() if len(pos) > 0 and len(neg) > 0 else np.nan

    return {"rho": rho, "gap": gap, "n": n}


def grid_search_universal(target_dfs, candidate_models, step=0.05, n_models=2):
    """
    여러 타겟의 평균 ρ를 최대화하는 universal 가중치 탐색.
    n_models개 모델 조합에 대해 grid search.
    """
    all_models = list(set(candidate_models))
    steps = int(1.0 / step) + 1
    weight_values = [round(i * step, 2) for i in range(steps)]

    best_avg_rho = -999
    best_result = None

    for combo in itertools.combinations(all_models, n_models):
        cols = list(combo)

        for w_combo in itertools.product(weight_values, repeat=n_models):
            if abs(sum(w_combo) - 1.0) > 0.001:
                continue
            if any(w == 0 for w in w_combo):
                continue

            rhos = []
            for target, df in target_dfs.items():
                # 모든 모델 컬럼이 있는지 확인
                missing = [m for m in cols if f"{m}_pKd" not in df.columns]
                if missing:
                    continue
                result = evaluate_recipe(df, cols, list(w_combo))
                if not np.isnan(result["rho"]):
                    rhos.append(result["rho"])

            if len(rhos) < len(target_dfs):
                continue

            avg_rho = np.mean(rhos)
            min_rho = np.min(rhos)

            if avg_rho > best_avg_rho:
                best_avg_rho = avg_rho
                best_result = {
                    "models": cols,
                    "weights": list(w_combo),
                    "avg_rho": avg_rho,
                    "min_rho": min_rho,
                    "rhos": dict(zip(target_dfs.keys(), rhos)),
                }

    return best_result


def main():
    report = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    log("=" * 90)
    log("  Phase 2.5: 앙상블 레시피 교차 검증")
    log("=" * 90)

    # ── 데이터 로드 ──
    target_dfs = {}
    for target in TARGETS:
        csv_path = os.path.join(VALIDATION_DIR, f"control_validation_{target}.csv")
        if os.path.exists(csv_path):
            target_dfs[target] = pd.read_csv(csv_path)

    log(f"\n  로드된 타겟: {list(target_dfs.keys())}")

    # ══════════════════════════════════════════════════════════
    # 1. 6×6 교차 매트릭스
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  1. 교차 매트릭스: 레시피(행) × 타겟(열) → Spearman ρ")
    log(f"{'=' * 90}")

    matrix_rho = {}
    matrix_gap = {}

    for recipe_name, recipe in PHASE2_RECIPES.items():
        matrix_rho[recipe_name] = {}
        matrix_gap[recipe_name] = {}
        for target, df in target_dfs.items():
            result = evaluate_recipe(df, recipe["models"], recipe["weights"])
            matrix_rho[recipe_name][target] = result["rho"]
            matrix_gap[recipe_name][target] = result["gap"]

    # 매트릭스 출력
    header = f"  {'레시피':<12}"
    for t in TARGETS:
        header += f" {t.upper():>8}"
    header += f" {'평균':>8} {'최소':>8}"
    log(f"\n{header}")
    log(f"  {'-'*12}" + f" {'-'*8}" * len(TARGETS) + f" {'-'*8} {'-'*8}")

    matrix_rows = []
    for recipe_name in TARGETS:
        row_vals = [matrix_rho[recipe_name].get(t, np.nan) for t in TARGETS]
        valid_vals = [v for v in row_vals if not np.isnan(v)]
        avg_rho = np.mean(valid_vals) if valid_vals else np.nan
        min_rho = np.min(valid_vals) if valid_vals else np.nan

        line = f"  {recipe_name:<12}"
        for t in TARGETS:
            val = matrix_rho[recipe_name].get(t, np.nan)
            if np.isnan(val):
                line += f" {'N/A':>8}"
            else:
                # 대각선(자기 자신) 강조
                marker = " ★" if recipe_name == t else ""
                line += f" {val:>7.3f}{marker}" if recipe_name != t else f"  {val:.3f}★"
            matrix_rows.append({
                "recipe": recipe_name, "target": t,
                "rho": val, "is_self": recipe_name == t
            })
        line += f" {avg_rho:>8.3f} {min_rho:>8.3f}"
        log(line)

    # ══════════════════════════════════════════════════════════
    # 2. 자기 타겟 vs 타 타겟 성능 비교
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  2. 타겟별 최적 레시피 vs 다른 레시피 적용 시 성능")
    log(f"{'=' * 90}")

    for target in TARGETS:
        df = target_dfs[target]
        log(f"\n  --- {target.upper()} ---")

        results = []
        for recipe_name, recipe in PHASE2_RECIPES.items():
            result = evaluate_recipe(df, recipe["models"], recipe["weights"])
            is_own = recipe_name == target
            results.append({
                "recipe": recipe_name,
                "rho": result["rho"],
                "gap": result["gap"],
                "is_own": is_own,
            })

        results.sort(key=lambda x: -(x["rho"] if not np.isnan(x["rho"]) else -999))

        log(f"    {'레시피':<12} {'ρ':>8} {'Gap':>8} {'비고'}")
        log(f"    {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
        for r in results:
            marker = "← 자기 레시피" if r["is_own"] else ""
            rho_str = f"{r['rho']:.3f}" if not np.isnan(r["rho"]) else "N/A"
            gap_str = f"{r['gap']:+.3f}" if not np.isnan(r["gap"]) else "N/A"
            log(f"    {r['recipe']:<12} {rho_str:>8} {gap_str:>8} {marker}")

        # 자기 레시피가 1등인지 확인
        own_rank = next(i+1 for i, r in enumerate(results) if r["is_own"])
        if own_rank == 1:
            log(f"    → 자기 레시피가 최적 ✅")
        else:
            best = results[0]
            log(f"    → {best['recipe']} 레시피가 더 나음 (ρ={best['rho']:.3f} vs 자기={results[own_rank-1]['rho']:.3f})")

    # ══════════════════════════════════════════════════════════
    # 3. Universal 후보 평가
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  3. Universal 레시피 후보 평가")
    log(f"{'=' * 90}")

    uni_results = []
    for name, recipe in UNIVERSAL_CANDIDATES.items():
        rhos = []
        for target, df in target_dfs.items():
            result = evaluate_recipe(df, recipe["models"], recipe["weights"])
            rhos.append(result["rho"] if not np.isnan(result["rho"]) else -999)

        avg_rho = np.mean([r for r in rhos if r > -999])
        min_rho = np.min([r for r in rhos if r > -999])
        uni_results.append({
            "name": name,
            "avg_rho": avg_rho,
            "min_rho": min_rho,
            "rhos": dict(zip(TARGETS, rhos)),
            "models": recipe["models"],
            "weights": recipe["weights"],
        })

    # 타겟별 최적 레시피의 평균도 비교용으로 추가
    own_rhos = [matrix_rho[t][t] for t in TARGETS if not np.isnan(matrix_rho[t].get(t, np.nan))]
    log(f"\n  참고: 타겟별 자기 레시피 평균 ρ = {np.mean(own_rhos):.3f}")

    uni_results.sort(key=lambda x: -x["avg_rho"])

    header = f"  {'후보':<42}"
    for t in TARGETS:
        header += f" {t.upper():>7}"
    header += f" {'평균':>7} {'최소':>7}"
    log(f"\n{header}")
    log(f"  {'-'*42}" + f" {'-'*7}" * len(TARGETS) + f" {'-'*7} {'-'*7}")

    for u in uni_results:
        line = f"  {u['name']:<42}"
        for t in TARGETS:
            val = u["rhos"].get(t, np.nan)
            line += f" {val:>7.3f}" if val > -999 else f" {'N/A':>7}"
        line += f" {u['avg_rho']:>7.3f} {u['min_rho']:>7.3f}"
        log(line)

    # ══════════════════════════════════════════════════════════
    # 4. Universal 최적 가중치 Grid Search (2모델, 3모델)
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  4. Universal 최적 가중치 Grid Search")
    log(f"{'=' * 90}")

    # 가장 출현 빈도 높은 모델들로 탐색
    top_candidates = ["PSICHIC", "GIGN", "MGraphDTA", "NMDN", "OnionNet2"]

    for n in [2, 3, 4]:
        log(f"\n  --- {n}모델 Universal 최적 ---")
        best = grid_search_universal(target_dfs, top_candidates, step=0.05, n_models=n)

        if best is None:
            log(f"    결과 없음")
            continue

        w_str = ", ".join([f"{m}={w:.2f}" for m, w in zip(best["models"], best["weights"])])
        log(f"    조합: {w_str}")
        log(f"    평균 ρ: {best['avg_rho']:.3f}, 최소 ρ: {best['min_rho']:.3f}")
        log(f"    타겟별:")
        for t in TARGETS:
            rho = best["rhos"].get(t, np.nan)
            log(f"      {t.upper():<8}: {rho:.3f}")

    # ══════════════════════════════════════════════════════════
    # 5. 최종 비교 및 추천
    # ══════════════════════════════════════════════════════════
    log(f"\n{'#' * 90}")
    log(f"  5. Phase 2.5 최종 비교")
    log(f"{'#' * 90}")

    log(f"\n  전략별 성능 비교:")
    log(f"    {'전략':<42} {'평균 ρ':>8} {'최소 ρ':>8} {'비고'}")
    log(f"    {'-'*42} {'-'*8} {'-'*8} {'-'*20}")

    # 타겟별 자기 레시피
    own_avg = np.mean(own_rhos)
    own_min = np.min(own_rhos)
    log(f"    {'타겟별 최적 레시피 (Phase 2)':<42} {own_avg:>8.3f} {own_min:>8.3f} {'타겟마다 다른 레시피'}")

    # Universal 후보들
    for u in uni_results[:3]:
        log(f"    {u['name']:<42} {u['avg_rho']:>8.3f} {u['min_rho']:>8.3f}")

    # 개별 최강 모델 단독
    solo_rhos = {}
    for target, df in target_dfs.items():
        best_solo = -999
        for m in ["PSICHIC", "GIGN", "MGraphDTA", "NMDN", "OnionNet2"]:
            col = f"{m}_pKd"
            if col not in df.columns:
                continue
            pred = df[col].values
            exp = df["Exp_pKd"].values
            valid = ~np.isnan(pred) & ~np.isnan(exp)
            if valid.sum() < 3:
                continue
            rho, _ = spearmanr(exp[valid], pred[valid])
            if rho > best_solo:
                best_solo = rho
        solo_rhos[target] = best_solo

    solo_avg = np.mean(list(solo_rhos.values()))
    solo_min = np.min(list(solo_rhos.values()))
    log(f"    {'타겟별 최고 단독 모델':<42} {solo_avg:>8.3f} {solo_min:>8.3f} {'기준선'}")

    log(f"\n  결론:")
    log(f"    1) 타겟별 최적 레시피 평균 ρ = {own_avg:.3f}")
    if uni_results:
        best_uni = uni_results[0]
        delta = own_avg - best_uni["avg_rho"]
        log(f"    2) 최고 Universal 후보 평균 ρ = {best_uni['avg_rho']:.3f} (차이: {delta:+.3f})")
        if delta < 0.05:
            log(f"    → Universal 레시피로 충분 (성능 차이 < 0.05)")
        else:
            log(f"    → 타겟별 레시피가 의미있는 향상 제공 (>{delta:.3f})")

    # ── 저장 ──
    report_path = os.path.join(OUTPUT_DIR, "phase2.5_cross_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n보고서 저장: {report_path}")

    # 교차 매트릭스 CSV
    matrix_df = pd.DataFrame(matrix_rho).T
    matrix_df.index.name = "recipe"
    matrix_path = os.path.join(OUTPUT_DIR, "phase2.5_cross_matrix.csv")
    matrix_df.to_csv(matrix_path)
    print(f"교차 매트릭스 저장: {matrix_path}")

    print("\n" + "=" * 90)
    print("  Phase 2.5 완료!")
    print("=" * 90)


if __name__ == "__main__":
    main()