#!/usr/bin/env python3
"""
bootstrap_ci.py — Spearman ρ Bootstrap 신뢰구간 계산
=====================================================
n=25와 n=110에서 각 앙상블 방법의 ρ에 대해
95% Bootstrap Confidence Interval을 계산합니다.

추가로 permutation test로 p-value도 계산합니다.

사용법:
  cd ~/Project/PLI
  python scripts/bootstrap_ci.py

출력:
  results/stacking/bootstrap_ci_report.txt
  results/stacking/bootstrap_ci.png
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR  = os.path.expanduser("~/Project/PLI")
RESULT_N25   = os.path.join(PROJECT_DIR, "results", "06_validation",
                             "control_validation_cdk2_n25_backup.csv")
RESULT_N110  = os.path.join(PROJECT_DIR, "results", "06_validation",
                             "control_validation_cdk2_n500.csv")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "results", "stacking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAP = 10000   # bootstrap 반복 수
N_PERMUTE   = 10000   # permutation 반복 수
ALPHA       = 0.05    # 유의수준 (95% CI)
RANDOM_SEED = 42

# Phase 2 레시피 (n=25 기반)
PHASE2_WEIGHTS = {
    "PSICHIC_pKd": 0.70, "NMDN_pKd": 0.05,
    "GIGN_pKd":    0.10, "PIGNet2_pKd": 0.15,
}

# n=110 최적 레시피
N110_WEIGHTS = {
    "GIGN_pKd":    0.50, "NMDN_pKd": 0.45,
    "DeepDTA_pKd": 0.05,
}

MODEL_COLS = ["PIGNet2_pKd","OnionNet2_pKd","DeepDTA_pKd","GraphDTA_pKd",
              "MGraphDTA_pKd","GIGN_pKd","PSICHIC_pKd","NMDN_pKd"]


# ═══════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    available = [c for c in MODEL_COLS if c in df.columns]
    valid_mask = df[available].notna().sum(axis=1) >= 3
    return df[valid_mask].reset_index(drop=True), available


def weighted_pred(df, weights):
    """가중 평균 예측"""
    preds = []
    for _, row in df.iterrows():
        vals, ws = [], []
        for col, w in weights.items():
            if w > 0 and col in df.columns and not pd.isna(row[col]):
                vals.append(float(row[col]))
                ws.append(w)
        preds.append(np.average(vals, weights=ws) if vals else np.nan)
    return np.array(preds)


def equal_weight_pred(df, available):
    """동일 가중 평균"""
    X = df[available].copy().fillna(df[available].mean()).fillna(0.0)
    return X.mean(axis=1).values


def calc_rho(y_true, y_pred):
    """유효한 쌍만으로 Spearman ρ 계산"""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 3:
        return np.nan
    return spearmanr(y_true[valid], y_pred[valid])[0]


# ═══════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=RANDOM_SEED):
    """
    Bootstrap 신뢰구간 계산
    반환: (observed_rho, lower, upper, se)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = calc_rho(y_true, y_pred)

    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = calc_rho(y_true[idx], y_pred[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    boot_rhos = np.array(boot_rhos)
    lower = np.percentile(boot_rhos, 100 * alpha / 2)
    upper = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
    se    = np.std(boot_rhos)

    return observed, lower, upper, se, boot_rhos


# ═══════════════════════════════════════════════════════════════
# Permutation Test
# ═══════════════════════════════════════════════════════════════

def permutation_test(y_true, y_pred, n_perm=N_PERMUTE, seed=RANDOM_SEED):
    """
    Permutation test: ρ가 우연인지 검정
    H0: 예측값과 실제값 사이에 상관관계 없음
    반환: (observed_rho, p_value)
    """
    rng = np.random.default_rng(seed)
    observed = calc_rho(y_true, y_pred)

    perm_rhos = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        r = calc_rho(y_perm, y_pred)
        if not np.isnan(r):
            perm_rhos.append(r)

    perm_rhos = np.array(perm_rhos)
    # 양측 검정: |perm_rho| >= |observed_rho|인 비율
    p_value = np.mean(np.abs(perm_rhos) >= np.abs(observed))

    return observed, p_value, perm_rhos


# ═══════════════════════════════════════════════════════════════
# 메인 실험
# ═══════════════════════════════════════════════════════════════

def run_bootstrap(label, csv_path, weights_dict):
    """한 데이터셋에 대해 여러 방법의 Bootstrap CI 계산"""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    if not os.path.exists(csv_path):
        print(f"  파일 없음: {csv_path}")
        return None

    df, available = load_data(csv_path)
    y_true = df["Exp_pKd"].values.astype(float)
    n = len(df)
    print(f"  n={n}  |  Bootstrap {N_BOOTSTRAP:,}회  |  Permutation {N_PERMUTE:,}회")

    results = []
    print(f"\n  {'방법':<28} {'ρ':>7}  {'95% CI':^18}  {'SE':>6}  {'p-value':>8}  {'유의?'}")
    print(f"  {'-'*28} {'-'*7}  {'-'*18}  {'-'*6}  {'-'*8}  {'-'*5}")

    for method_name, y_pred in weights_dict.items():
        if y_pred is None:
            continue

        # Bootstrap CI
        obs, lower, upper, se, boot_dist = bootstrap_ci(y_true, y_pred)

        # Permutation test
        _, p_val, perm_dist = permutation_test(y_true, y_pred)

        sig = "✅" if p_val < 0.05 else ("⚠️" if p_val < 0.10 else "❌")
        ci_str = f"[{lower:+.3f}, {upper:+.3f}]"

        print(f"  {method_name:<28} {obs:>+7.4f}  {ci_str:^18}  {se:>6.4f}  {p_val:>8.4f}  {sig}")

        results.append({
            "method": method_name,
            "rho": round(obs, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "se": round(se, 4),
            "p_value": round(p_val, 4),
            "significant": p_val < 0.05,
            "boot_dist": boot_dist,
            "perm_dist": perm_dist,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════

def plot_bootstrap(results_n25, results_n110):
    """Bootstrap 분포 + CI 시각화"""
    all_results = [("n=25", results_n25), ("n=110", results_n110)]
    n_methods = max(len(r) for _, r in all_results if r)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Bootstrap 95% CI — CDK2 앙상블 방법 비교",
                 fontsize=13, fontweight="bold")

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    for row_idx, (label, results) in enumerate(all_results):
        if not results:
            continue

        # 왼쪽: CI 플롯
        ax_ci = axes[row_idx][0]
        for i, r in enumerate(results):
            c = colors[i % len(colors)]
            ax_ci.errorbar(
                i, r["rho"],
                yerr=[[r["rho"] - r["ci_lower"]], [r["ci_upper"] - r["rho"]]],
                fmt="o", color=c, markersize=8, capsize=6, capthick=2, linewidth=2,
            )
            sig = "✅" if r["significant"] else "❌"
            ax_ci.text(i, r["ci_upper"] + 0.02,
                       f"p={r['p_value']:.3f}{sig}",
                       ha="center", fontsize=8, color=c)

        ax_ci.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_ci.set_xticks(range(len(results)))
        ax_ci.set_xticklabels([r["method"].replace("\n"," ") for r in results],
                               rotation=20, ha="right", fontsize=8)
        ax_ci.set_ylabel("Spearman ρ")
        ax_ci.set_title(f"{label} — 95% Bootstrap CI", fontweight="bold")
        ax_ci.grid(True, alpha=0.3, axis="y")

        # 오른쪽: Bootstrap 분포
        ax_dist = axes[row_idx][1]
        for i, r in enumerate(results):
            c = colors[i % len(colors)]
            ax_dist.hist(r["boot_dist"], bins=60, alpha=0.4, color=c,
                         label=r["method"].replace("\n"," "), density=True)
            ax_dist.axvline(r["rho"], color=c, linewidth=2, linestyle="-")
            ax_dist.axvline(r["ci_lower"], color=c, linewidth=1, linestyle="--", alpha=0.7)
            ax_dist.axvline(r["ci_upper"], color=c, linewidth=1, linestyle="--", alpha=0.7)

        ax_dist.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax_dist.set_xlabel("Spearman ρ")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title(f"{label} — Bootstrap 분포 (n_boot={N_BOOTSTRAP:,})",
                           fontweight="bold")
        ax_dist.legend(fontsize=7, loc="upper left")
        ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "bootstrap_ci.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")


# ═══════════════════════════════════════════════════════════════
# 보고서
# ═══════════════════════════════════════════════════════════════

def print_report(results_n25, results_n110):
    lines = []
    lines.append("=" * 70)
    lines.append("  Bootstrap 95% CI + Permutation Test 결과")
    lines.append(f"  Bootstrap: {N_BOOTSTRAP:,}회  |  Permutation: {N_PERMUTE:,}회")
    lines.append("=" * 70)

    for results, label in [(results_n25, "n=25"), (results_n110, "n=110")]:
        if not results:
            continue
        lines.append(f"\n[{label}]")
        lines.append(f"  {'방법':<28} {'ρ':>7}  {'95% CI':^18}  {'SE':>6}  {'p-value':>8}  {'유의?'}")
        lines.append(f"  {'-'*28} {'-'*7}  {'-'*18}  {'-'*6}  {'-'*8}  {'-'*5}")
        for r in results:
            sig = "✅ p<0.05" if r["significant"] else "❌ n.s."
            ci_str = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
            lines.append(f"  {r['method']:<28} {r['rho']:>+7.4f}  {ci_str:^18}  {r['se']:>6.4f}  {r['p_value']:>8.4f}  {sig}")

    lines.append(f"\n[해석]")
    lines.append(f"  - p < 0.05: 통계적으로 유의 (ρ가 우연이 아님)")
    lines.append(f"  - CI가 겹치지 않으면 두 방법 간 차이가 유의")
    lines.append(f"  - SE가 작을수록 ρ 추정이 안정적")

    # n=25 vs n=110 CI 비교
    if results_n25 and results_n110:
        lines.append(f"\n[n=25 vs n=110: 추정 안정성 비교]")
        lines.append(f"  SE가 클수록 n이 적어서 불안정함")
        for r25 in results_n25:
            for r110 in results_n110:
                if r25["method"] == r110["method"]:
                    lines.append(f"  {r25['method']:<28}: SE {r25['se']:.4f}(n=25) → {r110['se']:.4f}(n=110)  "
                                f"{'개선' if r110['se'] < r25['se'] else '악화'}")

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(OUTPUT_DIR, "bootstrap_ci_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  보고서 저장: {out}")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Bootstrap CI + Permutation Test")
    print(f"  Bootstrap {N_BOOTSTRAP:,}회 | Permutation {N_PERMUTE:,}회 | seed={RANDOM_SEED}")
    print("=" * 70)

    # n=25 데이터 준비
    df25, avail25 = load_data(RESULT_N25)
    y25 = df25["Exp_pKd"].values.astype(float)

    wa25 = weighted_pred(df25, PHASE2_WEIGHTS)
    eq25 = equal_weight_pred(df25, avail25)

    methods_n25 = {
        "Weighted Avg (Phase2)": wa25,
        "Equal Weight":          eq25,
    }
    # 개별 최고 모델 (PSICHIC)
    if "PSICHIC_pKd" in df25.columns:
        methods_n25["PSICHIC (best solo)"] = df25["PSICHIC_pKd"].fillna(
            df25["PSICHIC_pKd"].mean()).values

    # n=110 데이터 준비
    df110, avail110 = load_data(RESULT_N110)
    y110 = df110["Exp_pKd"].values.astype(float)

    wa110_phase2 = weighted_pred(df110, PHASE2_WEIGHTS)
    wa110_n110   = weighted_pred(df110, N110_WEIGHTS)
    eq110        = equal_weight_pred(df110, avail110)

    methods_n110 = {
        "Weighted Avg (Phase2)":  wa110_phase2,
        "Weighted Avg (n=110)":   wa110_n110,
        "Equal Weight":           eq110,
    }
    # 개별 최고 모델 (GIGN)
    if "GIGN_pKd" in df110.columns:
        methods_n110["GIGN (best solo)"] = df110["GIGN_pKd"].fillna(
            df110["GIGN_pKd"].mean()).values

    # Bootstrap 실행
    results_n25  = run_bootstrap("n=25  (기존 25개)",   RESULT_N25,  methods_n25)
    results_n110 = run_bootstrap("n=110 (확장 110개)", RESULT_N110, methods_n110)

    # 보고서
    print_report(results_n25, results_n110)

    # 시각화
    plot_bootstrap(results_n25, results_n110)

    print(f"\n{'='*70}")
    print(f"  완료! 결과: {OUTPUT_DIR}/")
    print(f"{'='*70}")
