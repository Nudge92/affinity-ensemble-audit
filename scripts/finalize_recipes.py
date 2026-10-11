#!/usr/bin/env python3
"""
finalize_recipes.py — Phase 3: 최종 앙상블 레시피 확정
================================================================
Phase 2 (타겟별 최적화) + Phase 2.5 (교차 검증) 결과를 종합하여
최종 앙상블 레시피를 확정합니다.

산출물:
  1) ensemble_recipes.json  — Snakefile/파이프라인에서 읽는 구조화된 레시피
  2) phase3_recipe_report.txt — 포트폴리오/발표용 근거 문서

사용법:
  cd ~/Project/PLI
  python scripts/finalize_recipes.py

입력:
  results/06_validation/control_validation_{target}.csv
  results/07_weighted_ensemble/phase2_target_recipes.csv
  results/07_weighted_ensemble/phase2.5_cross_matrix.csv
"""
import os
import json
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ─── 설정 ───
PROJECT_DIR = os.path.expanduser("~/Project/PLI")
VALIDATION_DIR = os.path.join(PROJECT_DIR, "results", "06_validation")
ENSEMBLE_DIR = os.path.join(PROJECT_DIR, "results", "07_weighted_ensemble")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "08_final_recipes")
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
TARGET_PDBS = {
    "cdk2": "cdk2_clean.pdb",
    "egfr": "4WKQ / egfr_clean.pdb",
    "hiv1pr": "3NU3 / hiv1pr_clean.pdb",
    "ptp1b": "2QBP / ptp1b_clean.pdb",
    "era": "3ERT / era_clean.pdb",
    "caii": "3HS4 / caii_clean.pdb",
}

ALL_MODELS = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA", "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]


# ═══════════════════════════════════════════════════════════════
# 확정 레시피 정의 (Phase 2 + 2.5 결과 기반)
# ═══════════════════════════════════════════════════════════════

FINAL_RECIPES = {
    # ── 타겟별 최적 레시피 (Phase 2 grid search 결과) ──
    "target_specific": {
        "cdk2": {
            "models": ["PSICHIC", "NMDN", "GIGN", "PIGNet2"],
            "weights": [0.70, 0.05, 0.10, 0.15],
            "ensemble_rho": 0.663,
            "best_solo": {"model": "PSICHIC", "rho": 0.546},
            "improvement": 0.117,
            "excluded": [],
            "notes": "8모델 전부 ρ>0, 동일가중(0.702)도 양호하나 PSICHIC 중심 4모델이 안정적",
        },
        "egfr": {
            "models": ["GraphDTA", "OnionNet2", "PSICHIC", "PIGNet2"],
            "weights": [0.40, 0.40, 0.10, 0.10],
            "ensemble_rho": 0.861,
            "best_solo": {"model": "MGraphDTA", "rho": 0.684},
            "improvement": 0.177,
            "excluded": ["DeepDTA"],
            "notes": "DeepDTA(ρ=-0.375) 제외. PSICHIC은 kinase domain(498aa)으로 실행 필수",
        },
        "hiv1pr": {
            "models": ["OnionNet2", "NMDN", "GIGN", "PSICHIC"],
            "weights": [0.45, 0.15, 0.15, 0.25],
            "ensemble_rho": 0.825,
            "best_solo": {"model": "OnionNet2", "rho": 0.768},
            "improvement": 0.057,
            "excluded": ["GraphDTA", "PIGNet2"],
            "notes": "PIGNet2는 dimer graph 생성 실패(N/A). GraphDTA(ρ=-0.005) 제외",
        },
        "ptp1b": {
            "models": ["MGraphDTA", "DeepDTA", "OnionNet2", "GraphDTA"],
            "weights": [0.40, 0.35, 0.05, 0.20],
            "ensemble_rho": 0.562,
            "best_solo": {"model": "MGraphDTA", "rho": 0.432},
            "improvement": 0.130,
            "excluded": ["PIGNet2"],
            "notes": "PIGNet2(ρ=-0.369) 제외. Davis 학습 모델(MGraphDTA, DeepDTA)이 강세",
        },
        "era": {
            "models": ["NMDN", "GIGN", "MGraphDTA", "PIGNet2"],
            "weights": [0.05, 0.10, 0.80, 0.05],
            "ensemble_rho": 0.664,
            "best_solo": {"model": "NMDN", "rho": 0.524},
            "improvement": 0.140,
            "excluded": ["DeepDTA", "GraphDTA"],
            "notes": "MGraphDTA 지배적(0.80). DeepDTA/GraphDTA 음수 ρ 제외",
        },
        "caii": {
            "models": ["PSICHIC", "GIGN", "MGraphDTA", "GraphDTA"],
            "weights": [0.30, 0.05, 0.30, 0.35],
            "ensemble_rho": 0.830,
            "best_solo": {"model": "PSICHIC", "rho": 0.688},
            "improvement": 0.141,
            "excluded": ["DeepDTA", "PIGNet2"],
            "notes": "DeepDTA/PIGNet2 음수 ρ 제외. GraphDTA가 단독 ρ=0.183이지만 조합에서 기여",
        },
    },

    # ── Universal fallback 레시피 ──
    "universal": {
        "fallback_3model": {
            "models": ["GIGN", "MGraphDTA", "NMDN"],
            "weights": [0.05, 0.80, 0.15],
            "avg_rho": 0.613,
            "min_rho": 0.440,
            "notes": "6타겟 평균 ρ 최대화 grid search 결과. 최소 ρ=0.440(PTP1B)",
        },
        "fallback_2model": {
            "models": ["MGraphDTA", "NMDN"],
            "weights": [0.85, 0.15],
            "avg_rho": 0.589,
            "min_rho": 0.423,
            "notes": "최소 모델 수 universal. MGraphDTA 지배적",
        },
        "fallback_safe": {
            "models": ["NMDN", "GIGN", "MGraphDTA", "PIGNet2"],
            "weights": [0.05, 0.10, 0.80, 0.05],
            "avg_rho": 0.588,
            "min_rho": 0.383,
            "notes": "ERα 레시피 = 교차검증에서 가장 범용적 (평균 0.588, 최소 0.383)",
        },
    },
}


def weighted_score(row, models, weights):
    """가중 평균 pKd"""
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
    """레시피 평가"""
    preds = df.apply(lambda row: weighted_score(row, models, weights), axis=1)
    valid = ~preds.isna() & ~df["Exp_pKd"].isna()
    if valid.sum() < 3:
        return {"rho": np.nan, "gap": np.nan, "n": 0}

    exp = df.loc[valid, "Exp_pKd"].values
    pred = preds[valid].values
    rho, _ = spearmanr(exp, pred)

    df_tmp = df[valid].copy()
    df_tmp["_pred"] = pred
    pos = df_tmp[df_tmp["Type"] == "positive"]["_pred"]
    neg = df_tmp[df_tmp["Type"] == "negative"]["_pred"]
    gap = pos.mean() - neg.mean() if len(pos) > 0 and len(neg) > 0 else np.nan

    return {"rho": round(rho, 3), "gap": round(gap, 3), "n": int(valid.sum())}


def get_solo_rhos(df):
    """개별 모델 ρ"""
    rhos = {}
    for m in ALL_MODELS:
        col = f"{m}_pKd"
        if col not in df.columns:
            continue
        pred = df[col].values
        exp = df["Exp_pKd"].values
        valid = ~np.isnan(pred) & ~np.isnan(exp)
        if valid.sum() < 3:
            continue
        rho, _ = spearmanr(exp[valid], pred[valid])
        rhos[m] = round(rho, 3)
    return rhos


def main():
    report = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    log("=" * 90)
    log("  Phase 3: 최종 앙상블 레시피 확정")
    log("=" * 90)

    # ── 데이터 로드 ──
    target_dfs = {}
    for target in TARGETS:
        csv_path = os.path.join(VALIDATION_DIR, f"control_validation_{target}.csv")
        if os.path.exists(csv_path):
            target_dfs[target] = pd.read_csv(csv_path)

    # ══════════════════════════════════════════════════════════
    # 1. 최종 레시피 검증
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  1. 타겟별 최종 레시피 검증")
    log(f"{'=' * 90}")

    verified_recipes = {}
    for target in TARGETS:
        recipe = FINAL_RECIPES["target_specific"][target]
        df = target_dfs[target]

        # 재검증
        result = evaluate_recipe(df, recipe["models"], recipe["weights"])
        solo_rhos = get_solo_rhos(df)
        best_solo = max(solo_rhos.items(), key=lambda x: x[1])

        log(f"\n  ── {target.upper()} ({TARGET_FAMILIES[target]}) ──")
        log(f"    PDB: {TARGET_PDBS[target]}")
        log(f"    레시피: {' + '.join(recipe['models'])}")
        log(f"    가중치: {' / '.join([str(w) for w in recipe['weights']])}")
        log(f"    앙상블 ρ = {result['rho']:.3f}  (검증 완료 ✅)")
        log(f"    최고 단독: {best_solo[0]} (ρ={best_solo[1]:.3f})")
        log(f"    향상: {result['rho'] - best_solo[1]:+.3f}")
        log(f"    Gap: {result['gap']:+.3f}")
        log(f"    제외: {', '.join(recipe['excluded']) if recipe['excluded'] else '없음'}")
        log(f"    비고: {recipe['notes']}")

        verified_recipes[target] = {
            "verified_rho": result["rho"],
            "verified_gap": result["gap"],
            "best_solo_model": best_solo[0],
            "best_solo_rho": best_solo[1],
        }

    # ══════════════════════════════════════════════════════════
    # 2. Universal fallback 검증
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  2. Universal Fallback 레시피 검증")
    log(f"{'=' * 90}")

    for name, recipe in FINAL_RECIPES["universal"].items():
        log(f"\n  ── {name} ──")
        log(f"    조합: {' + '.join([f'{m}({w})' for m, w in zip(recipe['models'], recipe['weights'])])}")

        rhos = []
        for target in TARGETS:
            df = target_dfs[target]
            result = evaluate_recipe(df, recipe["models"], recipe["weights"])
            rhos.append(result["rho"])
            log(f"    {target.upper():<8}: ρ={result['rho']:.3f}  gap={result['gap']:+.3f}")

        avg = np.mean(rhos)
        mn = np.min(rhos)
        log(f"    평균 ρ={avg:.3f}, 최소 ρ={mn:.3f}")

    # ══════════════════════════════════════════════════════════
    # 3. 전략 비교 종합표
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  3. 전략 비교 종합표")
    log(f"{'=' * 90}")

    header = f"\n  {'전략':<28}"
    for t in TARGETS:
        header += f" {t.upper():>7}"
    header += f" {'평균':>7} {'최소':>7}"
    log(header)
    log(f"  {'-'*28}" + f" {'-'*7}" * len(TARGETS) + f" {'-'*7} {'-'*7}")

    strategies = {}

    # a) 최고 단독 모델
    solo_vals = []
    line = f"  {'최고 단독 모델':<28}"
    for target in TARGETS:
        solo_rhos = get_solo_rhos(target_dfs[target])
        best = max(solo_rhos.values())
        solo_vals.append(best)
        line += f" {best:>7.3f}"
    line += f" {np.mean(solo_vals):>7.3f} {np.min(solo_vals):>7.3f}"
    log(line)
    strategies["최고 단독 모델"] = {"avg": np.mean(solo_vals), "min": np.min(solo_vals)}

    # b) 동일가중 전체 모델
    eq_vals = []
    line = f"  {'동일가중 전체 모델':<28}"
    for target in TARGETS:
        df = target_dfs[target]
        available = [m for m in ALL_MODELS if f"{m}_pKd" in df.columns]
        cols = [f"{m}_pKd" for m in available]
        w = [1.0 / len(available)] * len(available)
        result = evaluate_recipe(df, available, w)
        eq_vals.append(result["rho"])
        line += f" {result['rho']:>7.3f}"
    line += f" {np.mean(eq_vals):>7.3f} {np.min(eq_vals):>7.3f}"
    log(line)
    strategies["동일가중 전체"] = {"avg": np.mean(eq_vals), "min": np.min(eq_vals)}

    # c) Universal fallback (3모델)
    uni3 = FINAL_RECIPES["universal"]["fallback_3model"]
    uni3_vals = []
    line = f"  {'Universal 3모델':<28}"
    for target in TARGETS:
        result = evaluate_recipe(target_dfs[target], uni3["models"], uni3["weights"])
        uni3_vals.append(result["rho"])
        line += f" {result['rho']:>7.3f}"
    line += f" {np.mean(uni3_vals):>7.3f} {np.min(uni3_vals):>7.3f}"
    log(line)
    strategies["Universal 3모델"] = {"avg": np.mean(uni3_vals), "min": np.min(uni3_vals)}

    # d) 타겟별 최적 레시피
    opt_vals = []
    line = f"  {'★ 타겟별 최적 레시피':<28}"
    for target in TARGETS:
        recipe = FINAL_RECIPES["target_specific"][target]
        result = evaluate_recipe(target_dfs[target], recipe["models"], recipe["weights"])
        opt_vals.append(result["rho"])
        line += f" {result['rho']:>7.3f}"
    line += f" {np.mean(opt_vals):>7.3f} {np.min(opt_vals):>7.3f}"
    log(line)
    strategies["타겟별 최적 레시피"] = {"avg": np.mean(opt_vals), "min": np.min(opt_vals)}

    # ══════════════════════════════════════════════════════════
    # 4. 모델 기여도 분석
    # ══════════════════════════════════════════════════════════
    log(f"\n{'=' * 90}")
    log(f"  4. 모델 기여도 분석")
    log(f"{'=' * 90}")

    model_appearances = defaultdict(int)
    model_total_weight = defaultdict(float)
    model_targets = defaultdict(list)

    for target, recipe in FINAL_RECIPES["target_specific"].items():
        for m, w in zip(recipe["models"], recipe["weights"]):
            model_appearances[m] += 1
            model_total_weight[m] += w
            model_targets[m].append(f"{target}({w:.2f})")

    log(f"\n  {'모델':<12} {'출현':>4} {'총가중치':>8} {'평균가중치':>8} {'타겟별 역할'}")
    log(f"  {'-'*12} {'-'*4} {'-'*8} {'-'*8} {'-'*40}")
    for m in sorted(model_appearances.keys(), key=lambda x: -model_total_weight[x]):
        n = model_appearances[m]
        total_w = model_total_weight[m]
        avg_w = total_w / n
        targets_str = ", ".join(model_targets[m])
        log(f"  {m:<12} {n:>4} {total_w:>8.2f} {avg_w:>8.2f} {targets_str}")

    # 제외 분석
    log(f"\n  모델별 제외 현황:")
    for m in ALL_MODELS:
        excluded_from = []
        for target, recipe in FINAL_RECIPES["target_specific"].items():
            if m in recipe["excluded"]:
                solo_rhos = get_solo_rhos(target_dfs[target])
                rho = solo_rhos.get(m, "N/A")
                excluded_from.append(f"{target}(ρ={rho})")
        if excluded_from:
            log(f"    {m:<12}: 제외됨 ← {', '.join(excluded_from)}")

    # ══════════════════════════════════════════════════════════
    # 5. 운영 가이드라인
    # ══════════════════════════════════════════════════════════
    log(f"\n{'#' * 90}")
    log(f"  5. 운영 가이드라인 (Phase 4 Snakefile 입력)")
    log(f"{'#' * 90}")

    log(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  새로운 타겟이 들어왔을 때 의사결정 트리                                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Q1. 타겟의 단백질 패밀리를 아는가?                                       │
  │   ├─ YES → 해당 패밀리 레시피 사용                                      │
  │   │   ├─ Kinase       → CDK2 또는 EGFR 레시피 (서브타입에 따라)          │
  │   │   ├─ Protease     → HIV-1PR 레시피                               │
  │   │   ├─ Phosphatase  → PTP1B 레시피                                 │
  │   │   ├─ Nuc.Receptor → ERα 레시피                                   │
  │   │   └─ Metalloenzyme→ CA-II 레시피                                 │
  │   └─ NO → Universal fallback 사용                                    │
  │       └─ GIGN(0.05) + MGraphDTA(0.80) + NMDN(0.15)                 │
  │                                                                     │
  │  Q2. PSICHIC 실행 시 서열 길이 > 500aa?                                │
  │   ├─ YES → domain만 추출하여 실행                                      │
  │   └─ NO  → 전체 서열 사용                                             │
  │                                                                     │
  │  Q3. PIGNet2가 dimer 구조로 실패하는가?                                  │
  │   ├─ YES → 레시피에서 PIGNet2 제외, 나머지 가중치 재정규화                   │
  │   └─ NO  → 정상 진행                                                 │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # ══════════════════════════════════════════════════════════
    # 저장
    # ══════════════════════════════════════════════════════════

    # 1) JSON 레시피 파일 (Snakefile용)
    json_output = {
        "version": "1.0",
        "phase": "Phase 3 — Final Recipes",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "target_specific": {},
        "universal": {},
        "model_info": {
            "davis_trained": ["DeepDTA", "GraphDTA", "MGraphDTA"],
            "pdbbind_trained": ["PIGNet2", "OnionNet2", "GIGN", "PSICHIC", "NMDN"],
            "requires_docking": ["PIGNet2", "OnionNet2", "GIGN", "NMDN"],
            "sequence_only": ["DeepDTA", "GraphDTA", "MGraphDTA", "PSICHIC"],
        },
    }

    for target, recipe in FINAL_RECIPES["target_specific"].items():
        json_output["target_specific"][target] = {
            "family": TARGET_FAMILIES[target],
            "pdb": TARGET_PDBS[target],
            "models": recipe["models"],
            "weights": recipe["weights"],
            "ensemble_rho": recipe["ensemble_rho"],
            "excluded": recipe["excluded"],
            "notes": recipe["notes"],
        }

    for name, recipe in FINAL_RECIPES["universal"].items():
        json_output["universal"][name] = {
            "models": recipe["models"],
            "weights": recipe["weights"],
            "avg_rho": recipe.get("avg_rho"),
            "min_rho": recipe.get("min_rho"),
            "notes": recipe["notes"],
        }

    json_path = os.path.join(OUTPUT_DIR, "ensemble_recipes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    log(f"\n  레시피 JSON 저장: {json_path}")

    # 2) 보고서
    report_path = os.path.join(OUTPUT_DIR, "phase3_recipe_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"  보고서 저장: {report_path}")

    # 3) 요약 CSV
    summary_rows = []
    for target in TARGETS:
        recipe = FINAL_RECIPES["target_specific"][target]
        v = verified_recipes[target]
        summary_rows.append({
            "target": target,
            "family": TARGET_FAMILIES[target],
            "recipe_models": " + ".join(recipe["models"]),
            "recipe_weights": " / ".join([str(w) for w in recipe["weights"]]),
            "ensemble_rho": v["verified_rho"],
            "best_solo_model": v["best_solo_model"],
            "best_solo_rho": v["best_solo_rho"],
            "improvement": v["verified_rho"] - v["best_solo_rho"],
            "gap": v["verified_gap"],
            "excluded": ", ".join(recipe["excluded"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "phase3_recipe_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"  요약 CSV 저장: {summary_path}")

    log(f"\n{'=' * 90}")
    log(f"  Phase 3 완료! → Phase 4 (Snakefile 자동화)로 진행 가능")
    log(f"{'=' * 90}")


if __name__ == "__main__":
    main()