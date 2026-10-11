#!/usr/bin/env python3
"""
stacking_experiment.py — Weighted Average vs Stacking 비교 (CI 포함)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_DIR = os.path.expanduser("~/Project/PLI")
RESULT_N25  = os.path.join(PROJECT_DIR, "results", "06_validation", "control_validation_cdk2_n25_backup.csv")
RESULT_N110 = os.path.join(PROJECT_DIR, "results", "06_validation", "control_validation_cdk2_n500.csv")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "results", "stacking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_COLS = ["PIGNet2_pKd","OnionNet2_pKd","DeepDTA_pKd","GraphDTA_pKd",
              "MGraphDTA_pKd","GIGN_pKd","PSICHIC_pKd","NMDN_pKd"]

CDK2_WEIGHTS = {"PSICHIC_pKd":0.70,"NMDN_pKd":0.05,"GIGN_pKd":0.10,
                "PIGNet2_pKd":0.15,"OnionNet2_pKd":0.00,"DeepDTA_pKd":0.00,
                "GraphDTA_pKd":0.00,"MGraphDTA_pKd":0.00}

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    available = [c for c in MODEL_COLS if c in df.columns]
    valid_mask = df[available].notna().sum(axis=1) >= 3
    return df[valid_mask].reset_index(drop=True), available

def concordance_index(y_true, y_pred):
    ind = np.argsort(y_true)
    y_true, y_pred = y_true[ind], y_pred[ind]
    z, S = 0.0, 0.0
    for i in range(len(y_true)-1, 0, -1):
        for j in range(i-1, -1, -1):
            if y_true[i] > y_true[j]:
                z += 1
                u = y_pred[i] - y_pred[j]
                if u > 0: S += 1
                elif u == 0: S += 0.5
    return round(S/z, 4) if z > 0 else 0.5

def calc_metrics(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = np.array(y_true)[valid], np.array(y_pred)[valid]
    return {"spearman": round(spearmanr(yt,yp)[0],4),
            "pearson":  round(pearsonr(yt,yp)[0],4),
            "rmse":     round(np.sqrt(mean_squared_error(yt,yp)),4),
            "ci":       concordance_index(yt,yp),
            "n":        int(len(yt))}

def weighted_average_pred(df, available_cols, weights):
    preds = []
    for _, row in df.iterrows():
        vals, ws = [], []
        for col in available_cols:
            w = weights.get(col, 0)
            if w > 0 and not pd.isna(row[col]):
                vals.append(float(row[col])); ws.append(w)
        preds.append(np.average(vals, weights=ws) if vals else np.nan)
    return np.array(preds)

def impute_X(df, cols):
    X = df[cols].copy()
    X = X.fillna(X.mean()).fillna(0.0)
    return X.values.astype(float)

def loo_stacking(df, available_cols, model_class, model_kwargs=None, scale=True):
    if model_kwargs is None: model_kwargs = {}
    y_true = df["Exp_pKd"].values.astype(float)
    X_full = impute_X(df, available_cols)
    y_pred_loo = np.zeros(len(df))
    for train_idx, test_idx in LeaveOneOut().split(X_full):
        Xtr, Xte = X_full[train_idx], X_full[test_idx]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        model = model_class(**model_kwargs)
        model.fit(Xtr, y_true[train_idx])
        y_pred_loo[test_idx] = model.predict(Xte)
    return y_pred_loo

def run_experiment(label, csv_path):
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    if not os.path.exists(csv_path):
        print(f"  파일 없음: {csv_path}"); return None

    df, available = load_data(csv_path)
    y_true = df["Exp_pKd"].values.astype(float)
    print(f"  유효 샘플: {len(df)}개  |  양성: {sum(df['Type']=='positive')}  음성: {sum(df['Type']=='negative')}")
    print(f"  pKd 범위: {y_true.min():.2f} ~ {y_true.max():.2f}")

    results = []
    hdr = f"  {'방법':<28} {'ρ':>8}  {'CI':>8}  {'RMSE':>8}  {'Pearson':>8}"
    sep = f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}"

    # 개별 모델
    print(f"\n  [개별 모델]\n{hdr}\n{sep}")
    for col in available:
        valid = df[col].notna()
        if valid.sum() < 5: continue
        m = calc_metrics(y_true[valid], df[col][valid].values)
        name = col.replace("_pKd","")
        print(f"  {name:<28} {m['spearman']:>+8.4f}  {m['ci']:>8.4f}  {m['rmse']:>8.4f}  {m['pearson']:>+8.4f}")
        results.append({"method": name, "type": "solo", **m})

    # 앙상블
    print(f"\n  [앙상블]\n{hdr}\n{sep}")
    wa = weighted_average_pred(df, available, CDK2_WEIGHTS)
    valid = ~np.isnan(wa)
    m = calc_metrics(y_true[valid], wa[valid])
    print(f"  {'Weighted Avg (Phase2)':<28} {m['spearman']:>+8.4f}  {m['ci']:>8.4f}  {m['rmse']:>8.4f}  {m['pearson']:>+8.4f}")
    results.append({"method": "Weighted Average\n(Phase2 recipe)", "type": "ensemble", **m})

    eq = impute_X(df, available).mean(axis=1)
    m = calc_metrics(y_true, eq)
    print(f"  {'Equal Weight':<28} {m['spearman']:>+8.4f}  {m['ci']:>8.4f}  {m['rmse']:>8.4f}  {m['pearson']:>+8.4f}")
    results.append({"method": "Equal Weight\nAverage", "type": "ensemble", **m})

    # Stacking
    print(f"\n  [Stacking - LOO-CV]\n{hdr}\n{sep}")
    for name, cls, kwargs, scale in [
        ("Linear Stacking",  LinearRegression, {},                         True),
        ("Ridge (α=0.1)",    Ridge,            {"alpha": 0.1},             True),
        ("Ridge (α=1.0)",    Ridge,            {"alpha": 1.0},             True),
        ("Ridge (α=10.0)",   Ridge,            {"alpha": 10.0},            True),
    ]:
        pred = loo_stacking(df, available, cls, kwargs, scale)
        m = calc_metrics(y_true, pred)
        print(f"  {name:<28} {m['spearman']:>+8.4f}  {m['ci']:>8.4f}  {m['rmse']:>8.4f}  {m['pearson']:>+8.4f}")
        results.append({"method": name.replace(" ","_")+"\nStacking" if "Linear" in name else name.replace("Ridge","Ridge\n"), "type": "stacking", **m})

    if HAS_XGB:
        pred = loo_stacking(df, available, XGBRegressor,
                            {"n_estimators":50,"max_depth":3,"learning_rate":0.1,"verbosity":0}, False)
        m = calc_metrics(y_true, pred)
        print(f"  {'XGBoost':<28} {m['spearman']:>+8.4f}  {m['ci']:>8.4f}  {m['rmse']:>8.4f}  {m['pearson']:>+8.4f}")
        results.append({"method": "XGBoost\nStacking", "type": "stacking", **m})

    return pd.DataFrame(results)

def print_report(df25, df110):
    lines = ["="*72, "  CDK2 Stacking 실험 결과 보고서", "="*72]
    hdr = f"  {'방법':<28} {'ρ':>8}  {'CI':>8}  {'RMSE':>8}  {'Pearson':>8}"
    sep = f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}"

    for df, lbl in [(df25,"n=25 (기존)"), (df110,"n=110 (확장)")]:
        if df is None: continue
        lines.append(f"\n[{lbl}]\n{hdr}\n{sep}")
        sub = df[df["type"].isin(["ensemble","stacking"])].sort_values("spearman",ascending=False)
        for _, r in sub.iterrows():
            name = r["method"].replace("\n"," ")
            lines.append(f"  {name:<28} {r['spearman']:>+8.4f}  {r['ci']:>8.4f}  {r['rmse']:>8.4f}  {r['pearson']:>+8.4f}")

    if df25 is not None and df110 is not None:
        wa25  = df25[df25["method"].str.contains("Weighted",na=False)].iloc[0]
        wa110 = df110[df110["method"].str.contains("Weighted",na=False)].iloc[0]
        st25  = df25[df25["type"]=="stacking"].sort_values("spearman",ascending=False).iloc[0]
        st110 = df110[df110["type"]=="stacking"].sort_values("spearman",ascending=False).iloc[0]
        d25, d110 = st25["spearman"]-wa25["spearman"], st110["spearman"]-wa110["spearman"]

        lines += ["\n[핵심 비교]",
            f"\n  n=25:",
            f"    Weighted Average : ρ={wa25['spearman']:+.4f}  CI={wa25['ci']:.4f}  RMSE={wa25['rmse']:.4f}",
            f"    Best Stacking    : ρ={st25['spearman']:+.4f}  CI={st25['ci']:.4f}  RMSE={st25['rmse']:.4f}",
            f"    Δρ               : {d25:+.4f}",
            f"\n  n=110:",
            f"    Weighted Average : ρ={wa110['spearman']:+.4f}  CI={wa110['ci']:.4f}  RMSE={wa110['rmse']:.4f}",
            f"    Best Stacking    : ρ={st110['spearman']:+.4f}  CI={st110['ci']:.4f}  RMSE={st110['rmse']:.4f}",
            f"    Δρ               : {d110:+.4f}",
            "\n[해석]"]

        if d25 < 0 and d110 > 0:
            lines += ["  → n=25: Weighted Average 우위 (stacking 과적합)",
                      f"  → n=110: Stacking이 ρ 기준 {d110:+.4f}, RMSE 기준 {st110['rmse']-wa110['rmse']:+.4f} 향상",
                      "  → 데이터가 늘수록 meta-learner가 비선형 결합을 학습 가능"]
        elif d25 < 0 and d110 <= 0:
            lines += ["  → 두 경우 모두 Weighted Average 우위",
                      "  → CDK2에서는 단순 가중 평균이 더 robust"]
        else:
            lines += [f"  → n=110에서 stacking 경쟁력 있음 ({d110:+.4f})"]

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR,"stacking_report.txt"),"w",encoding="utf-8") as f:
        f.write(report)
    print(f"\n  보고서 저장: {os.path.join(OUTPUT_DIR,'stacking_report.txt')}")

def plot_comparison(df25, df110):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CDK2: Weighted Average vs Stacking\n(LOO-CV, n=25 vs n=110)", fontsize=14, fontweight="bold")
    colors = {"ensemble": "#3B82F6", "stacking": "#10B981", "solo": "#94A3B8"}
    metrics = [("spearman","Spearman ρ"),("ci","Concordance Index (CI)"),
               ("rmse","RMSE (낮을수록 좋음)"),("pearson","Pearson r")]

    for idx, (metric, mlabel) in enumerate(metrics):
        ax = axes[idx//2][idx%2]
        for di, (dlabel, df) in enumerate([("n=25",df25),("n=110",df110)]):
            if df is None: continue
            plot_df = df[df["type"].isin(["ensemble","stacking"])].copy()
            width = 0.12
            for bi, (_, row) in enumerate(plot_df.iterrows()):
                xpos = di + (bi - len(plot_df)/2) * width + width/2
                val = row[metric]
                c = colors.get(row["type"],"#94A3B8")
                ax.bar(xpos, val, width*0.9, color=c, alpha=0.85)
                ax.text(xpos, val+(0.005 if val>=0 else -0.02),
                        f"{val:.3f}", ha="center",
                        va="bottom" if val>=0 else "top", fontsize=6.5, rotation=90)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xticks([0,1]); ax.set_xticklabels(["n=25","n=110"], fontsize=11)
        ax.axhline(y=0, color="gray", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#3B82F6",label="Weighted/Equal Avg"),
                        Patch(color="#10B981",label="Stacking (LOO-CV)")],
               loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5,0.01))
    plt.tight_layout(rect=[0,0.05,1,1])
    out = os.path.join(OUTPUT_DIR,"stacking_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")

if __name__ == "__main__":
    print("="*72)
    print("  CDK2 Weighted Average vs Stacking (ρ / CI / RMSE / Pearson)")
    print("="*72)
    df25  = run_experiment("n=25  (기존 25개)",   RESULT_N25)
    df110 = run_experiment("n=110 (확장 110개)", RESULT_N110)
    all_results = pd.concat(
        [df.assign(dataset=lbl) for df,lbl in [(df25,"n=25"),(df110,"n=110")] if df is not None],
        ignore_index=True)
    all_results.to_csv(os.path.join(OUTPUT_DIR,"stacking_comparison.csv"), index=False)
    print_report(df25, df110)
    plot_comparison(df25, df110)
    print(f"\n{'='*72}\n  완료! 결과: {OUTPUT_DIR}/\n{'='*72}")
