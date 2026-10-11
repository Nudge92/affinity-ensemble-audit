#!/usr/bin/env python3
"""CDK2 451개에서 FEP(상대 결합 자유에너지)를 돌릴 만한 쌍을 찾는다.

조건
  - 같은 Murcko 골격
  - MCS 기준 한쪽에만 있는 중원자 5개 이내
  - ΔpKd >= 1.0 (1.5 이상 우선)
  - 회전가능결합 <= 8 (양쪽 모두)
  - 분자량 <= 500 (양쪽 모두)
"""
import itertools
import json
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SRC = "./results/10_baseline/cdk2_ic50_nothreshold.csv"
OUT = "./results/12_fep_pairs"

MAX_UNIQUE = 5      # 한쪽에만 있는 중원자 상한
MIN_DPKD = 1.0      # 활성 차이 하한
PREF_DPKD = 1.5     # 우선 기준
MAX_ROTB = 8
MAX_MW = 500.0


def prep(df):
    rows = []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r.SMILES)
        if m is None:
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        rows.append({
            "id": r.ID, "chembl": r.Drug_Name, "smiles": r.SMILES, "pkd": float(r.pKd),
            "mol": m, "n_heavy": m.GetNumHeavyAtoms(),
            "mw": Descriptors.MolWt(m), "rotb": Lipinski.NumRotatableBonds(m),
            "scaffold": Chem.MolToSmiles(scaf) if scaf.GetNumAtoms() else "",
        })
    return rows


def mcs_diff(a, b):
    """MCS 를 구해 (공통원자수, a고유, b고유) 반환. 실패하면 None."""
    res = rdFMCS.FindMCS(
        [a, b], timeout=10,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True, completeRingsOnly=True,
        matchValences=False)
    if res.canceled or res.numAtoms == 0:
        return None
    n = res.numAtoms
    return n, a.GetNumHeavyAtoms() - n, b.GetNumHeavyAtoms() - n


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SRC)
    rows = prep(df)
    print(f"분자 {len(rows)}개 파싱 (원본 {len(df)})")

    # 골격별 묶기
    by = {}
    for r in rows:
        if r["scaffold"]:
            by.setdefault(r["scaffold"], []).append(r)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    print(f"고유 Murcko 골격 {len(by)}개 · 2개 이상 공유 골격 {len(multi)}개 "
          f"(해당 분자 {sum(len(v) for v in multi.values())}개)")

    # 물성 필터를 먼저 적용해 후보를 줄인다
    def okprop(r):
        return r["mw"] <= MAX_MW and r["rotb"] <= MAX_ROTB
    n_prop = sum(1 for r in rows if okprop(r))
    print(f"물성 통과(MW<=500 & rotB<=8): {n_prop}/{len(rows)}")

    pairs, n_try = [], 0
    for scaf, members in multi.items():
        elig = [m for m in members if okprop(m)]
        for a, b in itertools.combinations(elig, 2):
            d = abs(a["pkd"] - b["pkd"])
            if d < MIN_DPKD:
                continue
            n_try += 1
            r = mcs_diff(a["mol"], b["mol"])
            if r is None:
                continue
            ncommon, ua, ub = r
            if max(ua, ub) > MAX_UNIQUE:
                continue
            hi, lo = (a, b) if a["pkd"] >= b["pkd"] else (b, a)
            pairs.append({
                "scaffold": scaf,
                "chembl_hi": hi["chembl"], "chembl_lo": lo["chembl"],
                "id_hi": hi["id"], "id_lo": lo["id"],
                "smiles_hi": hi["smiles"], "smiles_lo": lo["smiles"],
                "pkd_hi": round(hi["pkd"], 2), "pkd_lo": round(lo["pkd"], 2),
                "dpkd": round(d, 2),
                "mcs_atoms": ncommon,
                "uniq_hi": hi["n_heavy"] - ncommon, "uniq_lo": lo["n_heavy"] - ncommon,
                "uniq_max": max(ua, ub),
                "mw_hi": round(hi["mw"], 1), "mw_lo": round(lo["mw"], 1),
                "rotb_hi": hi["rotb"], "rotb_lo": lo["rotb"],
                "heavy_hi": hi["n_heavy"], "heavy_lo": lo["n_heavy"],
            })
    print(f"ΔpKd·물성 통과 후 MCS 계산 {n_try}쌍 → 최종 조건 통과 {len(pairs)}쌍")

    p = pd.DataFrame(pairs)
    if len(p):
        p = p.sort_values(["dpkd", "uniq_max"], ascending=[False, True]).reset_index(drop=True)
        p.insert(0, "rank", range(1, len(p) + 1))
        p["tier"] = ["우선(>=1.5)" if x >= PREF_DPKD else "포함(>=1.0)" for x in p.dpkd]
    p.to_csv(f"{OUT}/all_pairs.csv", index=False, encoding="utf-8")
    print(f"작성: {OUT}/all_pairs.csv ({len(p)}쌍)")
    if len(p):
        print(f"  ΔpKd>=1.5 : {(p.dpkd>=PREF_DPKD).sum()}쌍")
        print(f"  ΔpKd 1.0~1.5 : {(p.dpkd<PREF_DPKD).sum()}쌍")
        print(f"  고유 골격 : {p.scaffold.nunique()}개")
    json.dump({"n_mols": len(rows), "n_scaffolds": len(by),
               "n_shared_scaffolds": len(multi), "n_prop_ok": n_prop,
               "n_mcs_evaluated": n_try, "n_pairs": len(p),
               "params": {"MAX_UNIQUE": MAX_UNIQUE, "MIN_DPKD": MIN_DPKD,
                          "MAX_ROTB": MAX_ROTB, "MAX_MW": MAX_MW}},
              open(f"{OUT}/_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
