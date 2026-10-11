#!/usr/bin/env python3
"""
optimize_weights.py — 대조군 기반 앙상블 가중치 최적화
=========================================================
1. 대조군 검증 결과에서 최적 가중치를 grid search로 탐색
2. 최적 가중치를 실제 파이프라인 결과에 적용
3. 기존 동일 가중치 앙상블 vs 최적 가중치 앙상블 비교

사용법:
  cd ~/Project/PLI
  python scripts/optimize_weights.py

입력:
  results/06_validation/control_validation_results.csv
  results/03_affinity/*.csv

출력:
  results/07_weighted_ensemble/
"""
import os
import sys
import itertools

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

PROJECT_DIR = os.path.expanduser("~/Project/PLI")
VALIDATION_CSV = os.path.join(PROJECT_DIR, "results", "06_validation", "control_validation_results.csv")
AFFINITY_DIR = os.path.join(PROJECT_DIR, "results", "03_affinity")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "07_weighted_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA", "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]


# ═══════════════════════════════════════════════════════════════
# 1단계: Grid Search로 최적 가중치 탐색
# ═══════════════════════════════════════════════════════════════

def weighted_score(row, weights, model_cols):
    """가중 평균 pKd 계산"""
    vals = []
    ws = []
    for col, w in zip(model_cols, weights):
        v = row[col]
        if not np.isnan(v) and w > 0:
            vals.append(v)
            ws.append(w)
    if not ws:
        return np.nan
    return np.average(vals, weights=ws)


def evaluate_weights(df, weights, model_cols, target_col="Exp_pKd"):
    """특정 가중치 조합의 성능 평가"""
    df = df.copy()
    df["weighted_pred"] = df.apply(
        lambda row: weighted_score(row, weights, model_cols), axis=1
    )
    valid = df.dropna(subset=["weighted_pred", target_col])
    if len(valid) < 3:
        return {"spearman": -999, "pearson": -999, "rmse": 999}

    sp, _ = spearmanr(valid[target_col], valid["weighted_pred"])
    pe, _ = pearsonr(valid[target_col], valid["weighted_pred"])
    rmse = np.sqrt(np.mean((valid[target_col] - valid["weighted_pred"])**2))

    return {"spearman": sp, "pearson": pe, "rmse": rmse}


def grid_search_weights(df, model_cols, target_col="Exp_pKd", step=0.05):
    """
    Grid search로 최적 가중치 탐색.
    가중치 합 = 1.0, 각 가중치는 0.0~1.0 범위.
    """
    print("\n" + "=" * 70)
    print("  Grid Search: 최적 가중치 탐색")
    print("=" * 70)
    print(f"  모델: {model_cols}")
    print(f"  Step: {step}")

    best_spearman = -999
    best_weights = None
    best_metrics = None
    all_results = []

    # 가중치 조합 생성 (합 = 1.0)
    steps = int(1.0 / step) + 1
    weight_values = [round(i * step, 2) for i in range(steps)]

    n_combos = 0
    for w_combo in itertools.product(weight_values, repeat=len(model_cols)):
        if abs(sum(w_combo) - 1.0) > 0.001:
            continue
        n_combos += 1

        metrics = evaluate_weights(df, w_combo, model_cols, target_col)
        all_results.append({
            "weights": w_combo,
            "spearman": metrics["spearman"],
            "pearson": metrics["pearson"],
            "rmse": metrics["rmse"],
        })

        if metrics["spearman"] > best_spearman:
            best_spearman = metrics["spearman"]
            best_weights = w_combo
            best_metrics = metrics

    print(f"  탐색한 조합 수: {n_combos}")
    print(f"\n  최적 가중치:")
    for col, w in zip(model_cols, best_weights):
        model_name = col.replace("_pKd", "")
        print(f"    {model_name:<12}: {w:.2f}")
    print(f"\n  최적 성능:")
    print(f"    Spearman:  {best_metrics['spearman']:.3f}")
    print(f"    Pearson:   {best_metrics['pearson']:.3f}")
    print(f"    RMSE:      {best_metrics['rmse']:.3f}")

    # Top-10 조합 출력
    all_results.sort(key=lambda x: -x["spearman"])
    print(f"\n  Top-10 가중치 조합:")
    header = f"  {'순위':>4}"
    for col in model_cols:
        header += f" {col.replace('_pKd',''):>10}"
    header += f" {'Spearman':>10} {'Pearson':>9} {'RMSE':>8}"
    print(header)
    print(f"  {'-'*4}" + f" {'-'*10}" * len(model_cols) + f" {'-'*10} {'-'*9} {'-'*8}")
    for i, res in enumerate(all_results[:10]):
        w = res["weights"]
        line = f"  {i+1:>4}"
        for wv in w:
            line += f" {wv:>10.2f}"
        line += f" {res['spearman']:>10.3f} {res['pearson']:>9.3f} {res['rmse']:>8.3f}"
        print(line)

    return best_weights, best_metrics, all_results


def grid_search_2model_subsets(df, model_cols, target_col="Exp_pKd", step=0.05):
    """2모델 조합에 대해서도 최적 가중치 탐색"""
    print("\n" + "=" * 70)
    print("  2모델 부분 조합 최적 가중치")
    print("=" * 70)

    subset_results = {}
    for i in range(len(model_cols)):
        for j in range(i+1, len(model_cols)):
            cols_2 = [model_cols[i], model_cols[j]]
            name = "+".join([c.replace("_pKd", "") for c in cols_2])

            best_sp = -999
            best_w = None
            best_m = None

            steps = int(1.0 / step) + 1
            for w1_int in range(steps):
                w1 = round(w1_int * step, 2)
                w2 = round(1.0 - w1, 2)
                if w2 < 0:
                    continue
                metrics = evaluate_weights(df, (w1, w2), cols_2, target_col)
                if metrics["spearman"] > best_sp:
                    best_sp = metrics["spearman"]
                    best_w = (w1, w2)
                    best_m = metrics

            subset_results[name] = {
                "cols": cols_2,
                "weights": best_w,
                "metrics": best_m,
            }
            print(f"\n  [{name}]")
            print(f"    가중치: {cols_2[0].replace('_pKd','')}={best_w[0]:.2f}, "
                  f"{cols_2[1].replace('_pKd','')}={best_w[1]:.2f}")
            print(f"    Spearman={best_m['spearman']:.3f}, Pearson={best_m['pearson']:.3f}, "
                  f"RMSE={best_m['rmse']:.3f}")

    return subset_results


# ═══════════════════════════════════════════════════════════════
# 2단계: 실제 파이프라인 결과에 최적 가중치 적용
# ═══════════════════════════════════════════════════════════════

def apply_weights_to_pipeline(best_weights, model_names, gen="Pocket2Mol", admet_list=None):
    """실제 파이프라인 결과에 최적 가중치 적용"""
    if admet_list is None:
        admet_list = ["ADMET_AI", "Admetica"]

    print("\n" + "=" * 70)
    print("  실제 파이프라인 결과에 최적 가중치 적용")
    print("=" * 70)
    print(f"  가중치: {dict(zip(model_names, best_weights))}")

    for admet in admet_list:
        print(f"\n--- {admet} ---")

        # 각 모델 결과 로드
        model_dfs = {}
        for model in model_names:
            path = os.path.join(AFFINITY_DIR, f"{gen}-{admet}-{model}-scores.csv")
            if os.path.exists(path):
                mdf = pd.read_csv(path)
                if len(mdf) > 0:
                    model_dfs[model] = mdf
                    print(f"  {model}: {len(mdf)}개 분자 로드")

        if not model_dfs:
            print("  결과 없음!")
            continue

        # SMILES 기준으로 병합
        all_smiles = set()
        for mdf in model_dfs.values():
            all_smiles.update(mdf["SMILES"].tolist())

        rows = []
        for smi in all_smiles:
            row = {"SMILES": smi}
            scores = []
            ws = []

            for i, model in enumerate(model_names):
                if model in model_dfs:
                    match = model_dfs[model][model_dfs[model]["SMILES"] == smi]
                    if len(match) > 0:
                        score = float(match.iloc[0]["Affinity_Score"])
                        row[f"{model}_Score"] = score
                        row["ID"] = match.iloc[0].get("ID", "")
                        if best_weights[i] > 0:
                            scores.append(score)
                            ws.append(best_weights[i])

            if scores:
                row["Weighted_Score"] = np.average(scores, weights=ws)
            else:
                row["Weighted_Score"] = np.nan

            rows.append(row)

        result_df = pd.DataFrame(rows)
        result_df = result_df.dropna(subset=["Weighted_Score"])
        result_df["Weighted_Rank"] = result_df["Weighted_Score"].rank(ascending=False, method="min")
        result_df = result_df.sort_values("Weighted_Rank")

        # 기존 동일 가중치와 비교용
        equal_w = [1.0/len(model_names)] * len(model_names)
        for _, row_data in result_df.iterrows():
            eq_scores = []
            for model in model_names:
                col = f"{model}_Score"
                if col in row_data and not np.isnan(row_data[col]):
                    eq_scores.append(row_data[col])
            result_df.loc[result_df["SMILES"] == row_data["SMILES"], "Equal_Score"] = (
                np.mean(eq_scores) if eq_scores else np.nan
            )

        result_df["Equal_Rank"] = result_df["Equal_Score"].rank(ascending=False, method="min")

        # 출력
        print(f"\n  {'W_Rank':>6} {'E_Rank':>6} {'ID':<12} {'Weighted':>10} {'Equal':>10}", end="")
        for m in model_names:
            print(f" {m:>10}", end="")
        print()
        print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*10}", end="")
        for _ in model_names:
            print(f" {'-'*10}", end="")
        print()

        for _, r in result_df.head(20).iterrows():
            print(f"  {int(r['Weighted_Rank']):>6} {int(r.get('Equal_Rank', 0)):>6} "
                  f"{r.get('ID',''):>12} {r['Weighted_Score']:>10.3f} "
                  f"{r.get('Equal_Score', np.nan):>10.3f}", end="")
            for m in model_names:
                col = f"{m}_Score"
                val = r.get(col, np.nan)
                if not np.isnan(val):
                    print(f" {val:>10.3f}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            print()

        # 저장
        w_str = "_".join([f"{m}{w:.0f}" for m, w in zip(model_names, [x*100 for x in best_weights])])
        out_path = os.path.join(OUTPUT_DIR, f"{gen}-{admet}-weighted_{w_str}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"\n  저장: {out_path}")


# ═══════════════════════════════════════════════════════════════
# 3단계: 대조군에서의 가중 앙상블 상세 비교
# ═══════════════════════════════════════════════════════════════

def compare_ensemble_on_controls(df, best_weights_3, best_weights_2_on2_dta,
                                  model_cols_3, model_cols_2):
    """대조군에서 다양한 앙상블 전략 비교"""
    print("\n" + "=" * 70)
    print("  대조군에서 앙상블 전략 비교")
    print("=" * 70)

    strategies = {
        "동일가중 3모델": (model_cols_3, [1/3, 1/3, 1/3]),
        "최적가중 3모델": (model_cols_3, list(best_weights_3)),
        "OnionNet2 단독": (["OnionNet2_pKd"], [1.0]),
        "DeepDTA 단독": (["DeepDTA_pKd"], [1.0]),
        f"최적가중 ON2+DTA": (model_cols_2, list(best_weights_2_on2_dta)),
    }

    results = {}
    for name, (cols, weights) in strategies.items():
        df[f"pred_{name}"] = df.apply(
            lambda row: weighted_score(row, weights, cols), axis=1
        )
        valid = df.dropna(subset=[f"pred_{name}", "Exp_pKd"])
        if len(valid) >= 3:
            sp, _ = spearmanr(valid["Exp_pKd"], valid[f"pred_{name}"])
            pe, _ = pearsonr(valid["Exp_pKd"], valid[f"pred_{name}"])
            rmse = np.sqrt(np.mean((valid["Exp_pKd"] - valid[f"pred_{name}"])**2))

            # 양성/음성 분리
            pos = valid[valid["Type"] == "positive"][f"pred_{name}"]
            neg = valid[valid["Type"] == "negative"][f"pred_{name}"]
            gap = pos.mean() - neg.mean() if len(pos) > 0 and len(neg) > 0 else 0

            results[name] = {
                "spearman": sp, "pearson": pe, "rmse": rmse, "gap": gap
            }

    print(f"\n  {'전략':<25} {'Spearman':>10} {'Pearson':>9} {'RMSE':>8} {'양성-음성':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*9} {'-'*8} {'-'*10}")
    for name, m in results.items():
        gap_icon = "✅" if m["gap"] > 0.5 else "⚠️"
        print(f"  {name:<25} {m['spearman']:>10.3f} {m['pearson']:>9.3f} "
              f"{m['rmse']:>8.3f} {m['gap']:>+9.3f} {gap_icon}")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  앙상블 가중치 최적화")
    print("=" * 70)

    # 대조군 결과 로드
    df = pd.read_csv(VALIDATION_CSV)
    print(f"\n대조군 데이터: {len(df)}개 분자")

    # 사용 가능한 모델 컬럼 자동 탐색
    all_model_cols = [f"{m}_pKd" for m in MODELS if f"{m}_pKd" in df.columns]
    print(f"  사용 가능한 모델: {[c.replace('_pKd','') for c in all_model_cols]}")

    if len(all_model_cols) < 2:
        print("  ⚠️ 최소 2개 모델이 필요합니다!")
        return

    # ── 1) 2모델 부분 조합 Grid Search (항상 실행) ──
    subset_results = grid_search_2model_subsets(
        df, all_model_cols, step=0.05
    )

    # ── 2) 3모델 조합 Grid Search ──
    # 5모델 전체 grid search는 조합이 너무 많으므로 (step=0.05 → ~1M개)
    # 3모델 부분 조합으로 탐색
    if len(all_model_cols) >= 3:
        print("\n" + "=" * 70)
        print("  3모델 부분 조합 최적 가중치")
        print("=" * 70)

        best_3model = None
        best_3model_sp = -999

        from itertools import combinations
        for combo in combinations(range(len(all_model_cols)), 3):
            cols_3 = [all_model_cols[i] for i in combo]
            name_3 = "+".join([c.replace("_pKd", "") for c in cols_3])

            w3, m3, _ = grid_search_weights(df, cols_3, step=0.05)
            if m3["spearman"] > best_3model_sp:
                best_3model_sp = m3["spearman"]
                best_3model = {
                    "name": name_3,
                    "cols": cols_3,
                    "weights": w3,
                    "metrics": m3,
                }

        if best_3model:
            print(f"\n  전체 최고 3모델 조합: {best_3model['name']}")
            for c, w in zip(best_3model["cols"], best_3model["weights"]):
                print(f"    {c.replace('_pKd',''):<12}: {w:.2f}")
            print(f"    → Spearman={best_3model['metrics']['spearman']:.3f}")

    # ── 3) 대조군에서 전략 비교 ──
    print("\n" + "=" * 70)
    print("  대조군에서 앙상블 전략 비교")
    print("=" * 70)

    strategies = {}

    # 개별 모델
    for col in all_model_cols:
        name = col.replace("_pKd", "") + " 단독"
        strategies[name] = ([col], [1.0])

    # 동일 가중치 전체 모델
    n = len(all_model_cols)
    strategies[f"동일가중 {n}모델"] = (all_model_cols, [1.0/n] * n)

    # Top-2 조합
    if subset_results:
        best_2 = max(subset_results.values(), key=lambda x: x["metrics"]["spearman"])
        best_2_name = "+".join([c.replace("_pKd", "") for c in best_2["cols"]])
        strategies[f"최적가중 {best_2_name}"] = (best_2["cols"], list(best_2["weights"]))

    # Top-3 조합
    if len(all_model_cols) >= 3 and best_3model:
        strategies[f"최적가중 {best_3model['name']}"] = (
            best_3model["cols"], list(best_3model["weights"]))

    results = {}
    for name, (cols, weights) in strategies.items():
        df[f"pred_{name}"] = df.apply(
            lambda row: weighted_score(row, weights, cols), axis=1
        )
        valid = df.dropna(subset=[f"pred_{name}", "Exp_pKd"])
        if len(valid) >= 3:
            sp, _ = spearmanr(valid["Exp_pKd"], valid[f"pred_{name}"])
            pe, _ = pearsonr(valid["Exp_pKd"], valid[f"pred_{name}"])
            rmse_val = np.sqrt(np.mean((valid["Exp_pKd"] - valid[f"pred_{name}"])**2))
            pos = valid[valid["Type"] == "positive"][f"pred_{name}"]
            neg = valid[valid["Type"] == "negative"][f"pred_{name}"]
            gap = pos.mean() - neg.mean() if len(pos) > 0 and len(neg) > 0 else 0
            results[name] = {"spearman": sp, "pearson": pe, "rmse": rmse_val, "gap": gap}

    print(f"\n  {'전략':<35} {'Spearman':>10} {'Pearson':>9} {'RMSE':>8} {'양성-음성':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*9} {'-'*8} {'-'*10}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["spearman"]):
        gap_icon = "✅" if m["gap"] > 0.5 else "⚠️"
        print(f"  {name:<35} {m['spearman']:>10.3f} {m['pearson']:>9.3f} "
              f"{m['rmse']:>8.3f} {m['gap']:>+9.3f} {gap_icon}")

    # ── 4) 최적 가중치로 실제 파이프라인 결과에 적용 ──
    if subset_results:
        best_2 = max(subset_results.values(), key=lambda x: x["metrics"]["spearman"])
        model_names_2 = [c.replace("_pKd", "") for c in best_2["cols"]]
        print(f"\n\n{'#' * 70}")
        print(f"  최적 2모델 ({'+'.join(model_names_2)}) 가중치로 파이프라인 적용")
        print(f"{'#' * 70}")
        apply_weights_to_pipeline(best_2["weights"], model_names_2)

    if len(all_model_cols) >= 3 and best_3model:
        model_names_3 = [c.replace("_pKd", "") for c in best_3model["cols"]]
        print(f"\n\n{'#' * 70}")
        print(f"  최적 3모델 ({best_3model['name']}) 가중치로 파이프라인 적용")
        print(f"{'#' * 70}")
        apply_weights_to_pipeline(best_3model["weights"], model_names_3)

    # ── 5) 최종 요약 ──
    print("\n" + "=" * 70)
    print("  최종 요약")
    print("=" * 70)

    if subset_results:
        best_2 = max(subset_results.values(), key=lambda x: x["metrics"]["spearman"])
        best_2_name = "+".join([c.replace("_pKd", "") for c in best_2["cols"]])
        print(f"\n  최적 2모델 조합: {best_2_name}")
        for c, w in zip(best_2["cols"], best_2["weights"]):
            print(f"    {c.replace('_pKd',''):<12}: {w:.2f}")
        print(f"    → Spearman={best_2['metrics']['spearman']:.3f}")

    if len(all_model_cols) >= 3 and best_3model:
        print(f"\n  최적 3모델 조합: {best_3model['name']}")
        for c, w in zip(best_3model["cols"], best_3model["weights"]):
            print(f"    {c.replace('_pKd',''):<12}: {w:.2f}")
        print(f"    → Spearman={best_3model['metrics']['spearman']:.3f}")

    print(f"\n  결과 저장 경로: {OUTPUT_DIR}")
    print("=" * 70)
    print("  완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()