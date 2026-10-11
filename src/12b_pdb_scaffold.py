#!/usr/bin/env python3
"""쌍의 Murcko 골격이 CDK2 결정구조 리간드에 존재하는지 RCSB 에서 확인한다.

방법
  1) UniProt P24941(human CDK2) 에 연결된 PDB 엔트리를 검색
  2) 각 엔트리의 비폴리머(리간드) comp_id 수집
  3) comp_id 별 SMILES 를 받아 Murcko 골격 계산
  4) 우리 쌍의 골격과 정확히 일치하는지 대조
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
OUT = "./results/12_fep_pairs"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
SKIP = {"HOH", "GOL", "EDO", "SO4", "PO4", "CL", "NA", "MG", "ZN", "CA", "K",
        "ACT", "DMS", "PEG", "MPD", "TRS", "EPE", "IMD", "FMT", "NO3", "IOD",
        "BME", "CIT", "MES", "SIN", "ACY", "PG4", "1PE", "P6G", "MN", "NI", "CD"}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(1.5)
    return None


def cdk2_entries():
    q = {
        "query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers."
                         "reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": "P24941"}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 2000}},
    }
    url = ("https://search.rcsb.org/rcsbsearch/v2/query?json="
           + urllib.parse.quote(json.dumps(q)))
    d = get(url)
    if not d:
        return []
    return [x["identifier"] for x in d.get("result_set", [])]


def main():
    pairs = pd.read_csv(f"{OUT}/all_pairs.csv")
    want = sorted(set(pairs.scaffold))
    print(f"대조할 골격 {len(want)}개")

    ids = cdk2_entries()
    print(f"CDK2(P24941) PDB 엔트리 {len(ids)}개")
    if not ids:
        print("★검색 실패 — PDB 확인 없이 종료")
        json.dump({"error": "rcsb search failed"},
                  open(f"{OUT}/_pdb_scaffold.json", "w"), ensure_ascii=False)
        return

    # 엔트리별 리간드 comp_id (배치 GraphQL 로 한 번에)
    entry2lig, ligands = {}, set()
    B = 50
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        gql = ("{entries(entry_ids:[" + ",".join(f'"{x}"' for x in chunk) + "])"
               "{rcsb_id nonpolymer_entities{nonpolymer_comp{chem_comp{id}}}}}")
        url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(gql)
        d = get(url)
        if not d or "data" not in d:
            continue
        for e in d["data"].get("entries") or []:
            eid = e["rcsb_id"]
            ls = []
            for ne in (e.get("nonpolymer_entities") or []):
                cid = (((ne or {}).get("nonpolymer_comp") or {})
                       .get("chem_comp") or {}).get("id")
                if cid and cid not in SKIP:
                    ls.append(cid)
                    ligands.add(cid)
            entry2lig[eid] = ls
        print(f"  엔트리 {min(i+B,len(ids))}/{len(ids)} · 리간드 누적 {len(ligands)}", flush=True)

    print(f"고유 리간드 comp_id {len(ligands)}개 (물·이온·버퍼 제외)")

    # comp_id -> SMILES -> 골격
    lig2scaf = {}
    lids = sorted(ligands)
    for i in range(0, len(lids), B):
        chunk = lids[i:i + B]
        gql = ("{chem_comps(comp_ids:[" + ",".join(f'"{x}"' for x in chunk) + "])"
               "{chem_comp{id} rcsb_chem_comp_descriptor{SMILES_stereo SMILES}}}")
        url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(gql)
        d = get(url)
        if not d or "data" not in d:
            continue
        for c in d["data"].get("chem_comps") or []:
            cid = (c.get("chem_comp") or {}).get("id")
            desc = c.get("rcsb_chem_comp_descriptor") or {}
            smi = desc.get("SMILES_stereo") or desc.get("SMILES")
            if not (cid and smi):
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None or m.GetNumHeavyAtoms() < 8:
                continue
            s = MurckoScaffold.GetScaffoldForMol(m)
            if s.GetNumAtoms():
                lig2scaf[cid] = Chem.MolToSmiles(s)
        print(f"  리간드 {min(i+B,len(lids))}/{len(lids)} · 골격 확보 {len(lig2scaf)}", flush=True)

    # 골격 -> PDB ID 목록
    scaf2pdb = {}
    for eid, ls in entry2lig.items():
        for cid in ls:
            s = lig2scaf.get(cid)
            if s:
                scaf2pdb.setdefault(s, set()).add((eid, cid))
    hits = {w: sorted(scaf2pdb.get(w, set())) for w in want}
    n_hit = sum(1 for v in hits.values() if v)
    print(f"\n골격 {len(want)}개 중 CDK2 결정구조에 존재 {n_hit}개")

    json.dump({"n_cdk2_entries": len(ids), "n_ligands": len(ligands),
               "n_ligand_scaffolds": len(lig2scaf),
               "hits": {k: [{"pdb": p, "lig": l} for p, l in v] for k, v in hits.items()}},
              open(f"{OUT}/_pdb_scaffold.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def fmt(s):
        v = hits.get(s) or []
        if not v:
            return ""
        seen, out = set(), []
        for p, l in v:
            if l in seen:
                continue
            seen.add(l)
            out.append(f"{p}({l})")
            if len(out) >= 3:
                break
        return " ".join(out) + (f" 외 {len(v)-len(out)}" if len(v) > len(out) else "")

    pairs["pdb_scaffold"] = [fmt(s) for s in pairs.scaffold]
    pairs.to_csv(f"{OUT}/all_pairs.csv", index=False, encoding="utf-8")
    print(f"갱신: {OUT}/all_pairs.csv (pdb_scaffold 열 추가)")


if __name__ == "__main__":
    main()
