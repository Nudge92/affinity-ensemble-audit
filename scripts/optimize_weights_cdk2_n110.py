#!/usr/bin/env python3
"""
optimize_weights_cdk2_n110.py — CDK2 n=110 레시피 최적화
==========================================================
control_validation_cdk2_expanded.csv (n=110)으로
Phase 2와 동일한 방식으로 최적 가중치를 탐색합니다.

비교:
  - Phase 2 레시피 (n=25 기반): PSICHIC(0.70) + PIGNet2(0.15) + GIGN(0.10) + NMDN(0.05)
  - n=110 최적 레시피 (이 스크립트)

사용법:
  cd ~/Project/PLI
  python scripts/optimize_weights_cdk2_n110.py

출력:
  results/stacking/cdk2_n110_recipe.txt
  results/stacking/cdk2_n110_recipe.json
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR  = os.path.expanduser("~/Project/PLI")
INPUT_CSV    = os.path.join(PROJECT_DIR, "results", "06_validation",
                             "control_validation_cdk2_expanded.csv")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "results", "stacking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_MODELS = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA",
              "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]

GRID_STEP = 0.05   # 가중치 탐색 간격 (Phase 2와 동일)
MAX_MODELS = 4     # 최대 조합 모델 수

# Phase 2 기존 레시피 (n=25 기반)
PHASE2_RECIPE = {
    "PSICHIC": 0.70, "NMDN": 0.05,
    "GIGN": 0.10,    "PIGNet2": 0.15,
}


# ═══════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════

def get_model_rho(df):
    """각 모델의 개별 Spearman ρ 계산"""
    rhos = {}
    for m in ALL_MODELS:
        col = f"{m}_pKd"
        if col not in df.columns:
            continue
        valid = df[col].notna()
        if valid.sum() < 5:
            continue
        rho = spearmanr(df.loc[valid, "Exp_pKd"], df.loc[valid, col])[0]
        rhos[m] = round(rho, 4)
    return rhos


def weighted_score(row, models, weights):
    """가중 평균 pKd (NaN 모델 자동 스킵)"""
    vals, ws = [], []
    for m, w in zip(models, weights):
        col = f"{m}_pKd"
        v = row.get(col, np.nan)
        if not np.isnan(v) and w > 0:
            vals.append(v)
            ws.append(w)
    return np.average(vals, weights=ws) if vals else np.nan


def evaluate(df, models, weights):
    """특정 조합의 Spearman ρ 계산"""
    preds = df.apply(lambda r: weighted_score(r, models, weights), axis=1)
    valid = preds.notna() & df["Exp_pKd"].notna()
    if valid.sum() < 5:
        return -999
    return spearmanr(df.loc[valid, "Exp_pKd"], preds[valid])[0]


def grid_search(df, models):
    """N모델 grid search — 가중치 합=1.0, 모두 0 초과"""
    n = len(models)
    steps = int(1.0 / GRID_STEP) + 1
    weight_values = [round(i * GRID_STEP, 2) for i in range(steps)]

    best_rho, best_w = -999, None
    count = 0

    for w_combo in itertools.product(weight_values, repeat=n):
        if abs(sum(w_combo) - 1.0) > 0.001:
            continue
        if any(w == 0 for w in w_combo):
            continue  # 0인 모델은 사실상 n-1 조합이므로 스킵
        count += 1
        rho = evaluate(df, models, w_combo)
        if rho > best_rho:
            best_rho = rho
            best_w = w_combo

    return best_w, round(best_rho, 4), count


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  CDK2 n=110 레시피 최적화 (Phase 2 방식)")
    print("=" * 65)

    df = pd.read_csv(INPUT_CSV)
    print(f"\n  데이터: {len(df)}개 (양성 {sum(df['Type']=='positive')}, 음성 {sum(df['Type']=='negative')})")

    # 1. 개별 모델 ρ
    rhos = get_model_rho(df)
    print(f"\n  [개별 모델 Spearman ρ]")
    print(f"  {'모델':<14} {'ρ':>8}  {'판정'}")
    print(f"  {'-'*14} {'-'*8}  {'-'*10}")
    for m, r in sorted(rhos.items(), key=lambda x: -x[1]):
        status = "✅ pass" if r >= 0.3 else ("⚠️ weak" if r >= 0 else "❌ 제외")
        print(f"  {m:<14} {r:>+8.4f}  {status}")

    # 2. Phase 2 레시피 성능 (n=110에서)
    phase2_models = list(PHASE2_RECIPE.keys())
    phase2_weights = list(PHASE2_RECIPE.values())
    phase2_rho = evaluate(df, phase2_models, phase2_weights)
    print(f"\n  [Phase 2 레시피 성능 (n=25 기반, n=110 적용)]")
    print(f"  {' + '.join([f'{m}({w})' for m,w in PHASE2_RECIPE.items() if w > 0])}")
    print(f"  ρ = {phase2_rho:.4f}")

    # 3. ρ ≥ 0 모델만 후보
    eligible = {m: r for m, r in rhos.items() if r >= 0}
    excluded = [m for m, r in rhos.items() if r < 0]
    if excluded:
        print(f"\n  제외 모델 (ρ<0): {', '.join(excluded)}")

    eligible_sorted = sorted(eligible.items(), key=lambda x: -x[1])
    candidates = [m for m, _ in eligible_sorted]
    print(f"\n  후보 모델 ({len(candidates)}개): {candidates}")

    # 4. Grid Search (2~4모델 조합)
    print(f"\n  [Grid Search 진행 중... step={GRID_STEP}]")
    all_combos = []

    for n in range(2, min(MAX_MODELS + 1, len(candidates) + 1)):
        top_k = min(len(candidates), max(n + 2, 5))
        for combo in itertools.combinations(candidates[:top_k], n):
            best_w, best_rho, count = grid_search(df, list(combo))
            if best_w is not None:
                all_combos.append({
                    "models": combo,
                    "weights": best_w,
                    "rho": best_rho,
                    "n_models": n,
                })

    all_combos.sort(key=lambda x: -x["rho"])

    print(f"\n  [Top-10 조합]")
    print(f"  {'순위':>4}  {'조합':<45} {'ρ':>8}")
    print(f"  {'-'*4}  {'-'*45} {'-'*8}")
    for i, c in enumerate(all_combos[:10]):
        combo_str = " + ".join([f"{m}({w:.2f})"
                                 for m, w in zip(c["models"], c["weights"])])
        print(f"  {i+1:>4}  {combo_str:<45} {c['rho']:>+8.4f}")

    # 5. 최적 레시피 확정
    best = all_combos[0]
    best_models = list(best["models"])
    best_weights = list(best["weights"])
    best_rho = best["rho"]
    best_solo = eligible_sorted[0]

    print(f"\n  {'='*65}")
    print(f"  ★ n=110 최적 레시피")
    print(f"  {'='*65}")
    for m, w in zip(best_models, best_weights):
        print(f"    {m:<14}: {w:.2f}")
    print(f"\n    ρ (n=110):  {best_rho:+.4f}")
    print(f"    ρ (Phase2): {phase2_rho:+.4f}  (n=25 레시피를 n=110에 적용)")
    print(f"    향상:       {best_rho - phase2_rho:+.4f}")
    print(f"    vs 최고 단독 모델 ({best_solo[0]}, ρ={best_solo[1]:+.4f}): {best_rho - best_solo[1]:+.4f}")

    # 6. n=25 레시피와 비교
    print(f"\n  [Phase 2 레시피 vs n=110 레시피 비교]")
    print(f"  {'레시피':<35} {'기반 데이터':>12} {'n=110 ρ':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10}")
    print(f"  {'Phase2: ' + ' + '.join([f'{m}({w})' for m,w in PHASE2_RECIPE.items() if w > 0]):<35} {'n=25':>12} {phase2_rho:>+10.4f}")
    opt_str = ' + '.join([f'{m}({w:.2f})' for m, w in zip(best_models, best_weights)])
    print(f"  {'n=110: ' + opt_str:<35} {'n=110':>12} {best_rho:>+10.4f}")

    # 7. 저장
    recipe = {
        "n110_recipe": {
            "models": best_models,
            "weights": best_weights,
            "rho_n110": best_rho,
            "phase2_rho_on_n110": round(phase2_rho, 4),
            "improvement_vs_phase2": round(best_rho - phase2_rho, 4),
            "improvement_vs_solo": round(best_rho - best_solo[1], 4),
            "best_solo_model": best_solo[0],
            "best_solo_rho": best_solo[1],
            "excluded_models": excluded,
        }
    }
    json_path = os.path.join(OUTPUT_DIR, "cdk2_n110_recipe.json")
    with open(json_path, "w") as f:
        json.dump(recipe, f, indent=2)

    report_lines = [
        "=" * 65,
        "  CDK2 n=110 레시피 최적화 결과",
        "=" * 65,
        f"\n[개별 모델 ρ (n=110)]",
    ]
    for m, r in sorted(rhos.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {m:<14}: {r:+.4f}")
    report_lines += [
        f"\n[Phase 2 레시피 (n=25 기반) → n=110 적용]: ρ={phase2_rho:.4f}",
        f"\n[n=110 최적 레시피]",
        f"  모델: {' + '.join(best_models)}",
        f"  가중치: {' / '.join([f'{w:.2f}' for w in best_weights])}",
        f"  ρ (n=110): {best_rho:.4f}",
        f"  Phase2 대비 향상: {best_rho - phase2_rho:+.4f}",
        f"  단독 최고 모델 대비 향상: {best_rho - best_solo[1]:+.4f}",
    ]
    txt_path = os.path.join(OUTPUT_DIR, "cdk2_n110_recipe.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n  저장: {json_path}")
    print(f"  저장: {txt_path}")
    print(f"\n{'='*65}")
    print(f"  완료!")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
