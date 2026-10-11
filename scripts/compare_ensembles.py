"""
compare_ensembles.py — 7가지 앙상블 조합 비교
================================================
3개 모델(PIGNet2, OnionNet2, DeepDTA)의 모든 조합으로
앙상블 결과를 비교합니다.

조합:
  단일(3): PIGNet2, OnionNet2, DeepDTA
  2개(3):  PIGNet2+OnionNet2, PIGNet2+DeepDTA, OnionNet2+DeepDTA
  3개(1):  PIGNet2+OnionNet2+DeepDTA

사용법:
  python compare_ensembles.py --admet ADMET_AI
  python compare_ensembles.py --admet Admetica
  python compare_ensembles.py --admet all
"""
import os
import sys
import argparse
from itertools import combinations

import pandas as pd
import numpy as np


def load_affinity_results(affinity_dir, gen, admet, models):
    """각 모델의 affinity 결과를 로드"""
    results = {}
    for model in models:
        path = os.path.join(affinity_dir, f"{gen}-{admet}-{model}-scores.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0 and "Affinity_Score" in df.columns:
                df["Affinity_Score"] = pd.to_numeric(df["Affinity_Score"], errors="coerce")
                df = df.dropna(subset=["Affinity_Score"])
                if len(df) > 0:
                    results[model] = df
                    print(f"  ✅ {model}: {len(df)}개 분자, "
                          f"스코어 범위 [{df['Affinity_Score'].min():.3f}, {df['Affinity_Score'].max():.3f}]")
                else:
                    print(f"  ⚠️  {model}: 유효한 스코어 없음")
            else:
                print(f"  ⚠️  {model}: 빈 결과")
        else:
            print(f"  ❌ {model}: 파일 없음 ({path})")
    return results


def rank_within_model(df):
    """모델 내에서 percentile rank 산출 (낮을수록 좋음)"""
    df = df.copy()
    df["Rank"] = df["Affinity_Score"].rank(method="min", ascending=True)
    df["Percentile"] = df["Rank"] / len(df)
    return df


def ensemble_combination(results, model_names):
    """특정 모델 조합으로 앙상블 랭킹 생성"""
    ranked_dfs = []
    for model in model_names:
        if model not in results:
            return None
        ranked = rank_within_model(results[model])
        ranked_dfs.append(ranked)

    combined = pd.concat(ranked_dfs, ignore_index=True)

    agg = combined.groupby("SMILES").agg(
        Mean_Percentile=("Percentile", "mean"),
        Mean_Score=("Affinity_Score", "mean"),
        Min_Score=("Affinity_Score", "min"),
        Max_Score=("Affinity_Score", "max"),
        Std_Score=("Affinity_Score", "std"),
        Num_Models=("Affinity_Model", "nunique"),
        Models=("Affinity_Model", lambda x: "|".join(sorted(x.unique()))),
    ).reset_index()

    agg["Std_Score"] = agg["Std_Score"].fillna(0)

    # ID 복원
    id_map = combined.groupby("SMILES")["ID"].first().reset_index()
    agg = agg.merge(id_map, on="SMILES", how="left")

    # 랭킹
    agg["Ensemble_Rank"] = agg["Mean_Percentile"].rank(method="min")
    agg = agg.sort_values("Ensemble_Rank")

    return agg


def compare_all_combinations(results, models, top_n=10):
    """모든 조합을 비교하고 결과 반환"""
    all_combos = []

    # 1개, 2개, 3개 조합
    for r in range(1, len(models) + 1):
        for combo in combinations(models, r):
            combo_name = "+".join(combo)
            ensemble_df = ensemble_combination(results, list(combo))
            if ensemble_df is not None:
                all_combos.append({
                    "combo_name": combo_name,
                    "combo_size": r,
                    "models": list(combo),
                    "df": ensemble_df,
                    "n_molecules": len(ensemble_df),
                })

    return all_combos


def print_comparison_table(all_combos, top_n=10):
    """비교 결과를 표로 출력"""

    print("\n" + "=" * 100)
    print(f"  7가지 앙상블 조합 비교 — Top {top_n} 후보")
    print("=" * 100)

    # 1) 각 조합의 Top-N 리스트
    for combo in all_combos:
        name = combo["combo_name"]
        df = combo["df"]
        size = combo["combo_size"]
        label = {1: "단일", 2: "2개 조합", 3: "3개 조합"}[size]

        print(f"\n{'─' * 80}")
        print(f"  [{label}] {name}  (총 {combo['n_molecules']}개 분자)")
        print(f"{'─' * 80}")
        print(f"  {'순위':>4}  {'ID':<12} {'Mean_Score':>12} {'Min':>10} {'Max':>10} {'Std':>10}")
        print(f"  {'─'*4}  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

        for _, row in df.head(top_n).iterrows():
            print(f"  {int(row['Ensemble_Rank']):>4}  {row['ID']:<12} "
                  f"{row['Mean_Score']:>12.3f} {row['Min_Score']:>10.3f} "
                  f"{row['Max_Score']:>10.3f} {row['Std_Score']:>10.3f}")

    # 2) Top-1 비교 요약
    print(f"\n{'=' * 100}")
    print("  Top-1 후보 비교 요약")
    print(f"{'=' * 100}")
    print(f"  {'조합':<35} {'#분자':>5} {'Top-1 ID':<12} {'Mean Score':>12} {'Top-1 SMILES':<50}")
    print(f"  {'─'*35} {'─'*5} {'─'*12} {'─'*12} {'─'*50}")

    for combo in all_combos:
        name = combo["combo_name"]
        df = combo["df"]
        top1 = df.iloc[0]
        smi = top1["SMILES"][:47] + "..." if len(top1["SMILES"]) > 47 else top1["SMILES"]
        print(f"  {name:<35} {combo['n_molecules']:>5} {top1['ID']:<12} "
              f"{top1['Mean_Score']:>12.3f} {smi:<50}")

    # 3) Top-5 겹침 분석
    print(f"\n{'=' * 100}")
    print("  Top-5 겹침 분석 (어떤 분자가 여러 조합에서 공통으로 Top-5에 드는지)")
    print(f"{'=' * 100}")

    top5_sets = {}
    for combo in all_combos:
        top5_ids = set(combo["df"].head(5)["ID"].tolist())
        top5_sets[combo["combo_name"]] = top5_ids

    # 모든 Top-5에 나온 분자 수집
    all_top5_ids = set()
    for ids in top5_sets.values():
        all_top5_ids.update(ids)

    print(f"\n  {'분자 ID':<12}", end="")
    for combo in all_combos:
        short = combo["combo_name"].replace("OnionNet2", "ON2").replace("PIGNet2", "PIG").replace("DeepDTA", "DTA")
        print(f" {short:>12}", end="")
    print()

    print(f"  {'─'*12}", end="")
    for _ in all_combos:
        print(f" {'─'*12}", end="")
    print()

    for mol_id in sorted(all_top5_ids):
        print(f"  {mol_id:<12}", end="")
        for combo in all_combos:
            if mol_id in top5_sets[combo["combo_name"]]:
                # 해당 분자의 순위 찾기
                rank = combo["df"][combo["df"]["ID"] == mol_id]["Ensemble_Rank"].values
                rank_str = f"#{int(rank[0])}" if len(rank) > 0 else "?"
                print(f" {rank_str:>12}", end="")
            else:
                print(f" {'─':>12}", end="")
        print()

    # 4) Spearman 순위 상관
    print(f"\n{'=' * 100}")
    print("  Spearman 순위 상관 (조합 간 랭킹 일치도)")
    print(f"{'=' * 100}")

    from scipy.stats import spearmanr

    combo_names = [c["combo_name"] for c in all_combos]
    n = len(combo_names)

    # 공통 분자 기준 상관 계산
    all_smiles = set()
    for combo in all_combos:
        all_smiles.update(combo["df"]["SMILES"].tolist())

    rank_matrix = {}
    for combo in all_combos:
        rank_dict = dict(zip(combo["df"]["SMILES"], combo["df"]["Ensemble_Rank"]))
        rank_matrix[combo["combo_name"]] = rank_dict

    print(f"\n  {'':>20}", end="")
    for name in combo_names:
        short = name.replace("OnionNet2", "ON2").replace("PIGNet2", "PIG").replace("DeepDTA", "DTA")
        print(f" {short:>12}", end="")
    print()

    for i, name_i in enumerate(combo_names):
        short_i = name_i.replace("OnionNet2", "ON2").replace("PIGNet2", "PIG").replace("DeepDTA", "DTA")
        print(f"  {short_i:>20}", end="")
        for j, name_j in enumerate(combo_names):
            common = set(rank_matrix[name_i].keys()) & set(rank_matrix[name_j].keys())
            if len(common) >= 3:
                ranks_i = [rank_matrix[name_i][s] for s in common]
                ranks_j = [rank_matrix[name_j][s] for s in common]
                corr, _ = spearmanr(ranks_i, ranks_j)
                print(f" {corr:>12.3f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()


def save_results(all_combos, output_dir, gen, admet):
    """모든 조합 결과를 CSV로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    for combo in all_combos:
        name = combo["combo_name"].replace("+", "_")
        path = os.path.join(output_dir, f"{gen}-{admet}-ensemble_{name}.csv")
        combo["df"].to_csv(path, index=False)

    # 요약 CSV
    summary_rows = []
    for combo in all_combos:
        df = combo["df"]
        for _, row in df.iterrows():
            summary_rows.append({
                "Ensemble": combo["combo_name"],
                "Combo_Size": combo["combo_size"],
                "Ensemble_Rank": row["Ensemble_Rank"],
                "ID": row["ID"],
                "SMILES": row["SMILES"],
                "Mean_Percentile": row["Mean_Percentile"],
                "Mean_Score": row["Mean_Score"],
                "Min_Score": row["Min_Score"],
                "Max_Score": row["Max_Score"],
                "Std_Score": row["Std_Score"],
                "Num_Models": row.get("Num_Models", 1),
                "Models": row.get("Models", combo["combo_name"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"{gen}-{admet}-all_ensembles_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  📁 요약 저장: {summary_path}")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="7가지 앙상블 조합 비교")
    parser.add_argument("--admet", default="all", help="ADMET 모델 (ADMET_AI, Admetica, all)")
    parser.add_argument("--gen", default="Pocket2Mol", help="생성 모델")
    parser.add_argument("--affinity-dir", default="results/03_affinity", help="affinity 결과 디렉토리")
    parser.add_argument("--output-dir", default="results/05_ensemble_compare", help="비교 결과 출력 디렉토리")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N 후보 수")
    args = parser.parse_args()

    models = ["PIGNet2", "OnionNet2", "DeepDTA"]

    admet_list = ["ADMET_AI", "Admetica"] if args.admet == "all" else [args.admet]

    for admet in admet_list:
        print(f"\n{'#' * 100}")
        print(f"  ADMET 필터: {admet}")
        print(f"{'#' * 100}")

        print(f"\n모델별 결과 로드:")
        results = load_affinity_results(args.affinity_dir, args.gen, admet, models)

        if len(results) == 0:
            print("  ❌ 유효한 결과가 없습니다.")
            continue

        available_models = list(results.keys())
        print(f"\n사용 가능한 모델: {available_models}")

        all_combos = compare_all_combinations(results, available_models, args.top_n)
        print_comparison_table(all_combos, args.top_n)
        save_results(all_combos, args.output_dir, args.gen, admet)

    print(f"\n{'=' * 100}")
    print("  완료!")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()