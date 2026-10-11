#!/usr/bin/env python3
"""EGFR PSICHIC만 개별 실행 → 기존 CSV에 merge"""
import pandas as pd
import subprocess
import os
import tempfile

PROJECT_DIR = os.path.expanduser("~/Project/PLI")
PSICHIC_DIR = os.path.join(PROJECT_DIR, "external_models/PSICHIC")
PYTHON = os.path.expanduser("~/miniforge3/envs/psichic_fp/bin/python")

# EGFR kinase domain only (696-1022, 약 327aa → 안전하게 498aa 범위)
# UniProt P00533, kinase domain residues 696-1022
EGFR_KINASE_SEQ = (
    "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDN"
    "PHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDR"
    "RLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIY"
    "THQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCW"
    "MIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVV"
    "DADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSS"
    "DPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHS"
    "TAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGS"
    "TAENAEYLRVAPQSSEFIGA"
)

df = pd.read_csv(os.path.join(PROJECT_DIR, "data/control_ligands_egfr.csv"))

with tempfile.TemporaryDirectory() as tmpdir:
    # screening CSV 생성
    screen_csv = os.path.join(tmpdir, "psichic_egfr.csv")
    screen_df = pd.DataFrame({
        "Protein": [EGFR_KINASE_SEQ] * len(df),
        "Ligand": df["SMILES"].tolist(),
    })
    screen_df.to_csv(screen_csv, index=False)
    
    result_dir = os.path.join(tmpdir, "psichic_result")
    
    r = subprocess.run(
        [PYTHON, os.path.join(PSICHIC_DIR, "screening.py"),
         "--screenfile", screen_csv,
         "--trained_model_path", os.path.join(PSICHIC_DIR, "trained_weights/PDBv2020_PSICHIC"),
         "--result_path", result_dir,
         "--device", "cuda:0"],
        capture_output=True, text=True,
        cwd=PSICHIC_DIR,
    )
    
    if r.returncode != 0:
        print(f"FAIL: {r.stderr[-500:]}")
        exit(1)
    
    result_csv = os.path.join(result_dir, "screening.csv")
    if not os.path.exists(result_csv):
        print("결과 파일 없음")
        exit(1)
    
    pred_df = pd.read_csv(result_csv)
    psichic_map = {}
    for i, row in pred_df.iterrows():
        if i < len(df):
            psichic_map[df.iloc[i]["ID"]] = round(float(row["predicted_binding_affinity"]), 3)
    
    print(f"PSICHIC 성공: {len(psichic_map)}/{len(df)}개")
    for k, v in psichic_map.items():
        print(f"  {k}: {v}")
    
    # 기존 CSV에 merge
    out_csv = os.path.join(PROJECT_DIR, "results/06_validation/control_validation_egfr.csv")
    main_df = pd.read_csv(out_csv)
    main_df["PSICHIC_pKd"] = main_df["ID"].map(psichic_map)
    main_df.to_csv(out_csv, index=False)
    print(f"\nMerged → {out_csv}")

