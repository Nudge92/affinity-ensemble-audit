#!/usr/bin/env python3
"""HIV-1PR PSICHIC만 개별 실행 → 기존 CSV에 merge"""
import pandas as pd
import subprocess
import os
import tempfile

PROJECT_DIR = os.path.expanduser("~/Project/PLI")
PSICHIC_DIR = os.path.join(PROJECT_DIR, "external_models/PSICHIC")
PYTHON = os.path.expanduser("~/miniforge3/envs/psichic_fp/bin/python")

HIV1PR_SEQ = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

df = pd.read_csv(os.path.join(PROJECT_DIR, "data/control_ligands_hiv1pr.csv"))

# 개별 실행 (특정 SMILES 파싱 실패 대비)
psichic_map = {}
for idx, row in df.iterrows():
    mol_id = row["ID"]
    smi = row["SMILES"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        screen_csv = os.path.join(tmpdir, "screen.csv")
        pd.DataFrame({"Protein": [HIV1PR_SEQ], "Ligand": [smi]}).to_csv(screen_csv, index=False)
        
        result_dir = os.path.join(tmpdir, "result")
        r = subprocess.run(
            [PYTHON, os.path.join(PSICHIC_DIR, "screening.py"),
             "--screenfile", screen_csv,
             "--trained_model_path", os.path.join(PSICHIC_DIR, "trained_weights/PDBv2020_PSICHIC"),
             "--result_path", result_dir,
             "--device", "cuda:0"],
            capture_output=True, text=True,
            cwd=PSICHIC_DIR,
        )
        
        result_csv = os.path.join(result_dir, "screening.csv")
        if os.path.exists(result_csv):
            pred_df = pd.read_csv(result_csv)
            if "predicted_binding_affinity" in pred_df.columns and len(pred_df) > 0:
                val = round(float(pred_df.iloc[0]["predicted_binding_affinity"]), 3)
                psichic_map[mol_id] = val
                print(f"  {mol_id}: {val}")
                continue
        print(f"  {mol_id}: FAIL")

print(f"\nPSICHIC 성공: {len(psichic_map)}/{len(df)}개")

# 기존 CSV에 merge
out_csv = os.path.join(PROJECT_DIR, "results/06_validation/control_validation_hiv1pr.csv")
main_df = pd.read_csv(out_csv)
main_df["PSICHIC_pKd"] = main_df["ID"].map(psichic_map)
main_df.to_csv(out_csv, index=False)
print(f"Merged → {out_csv}")

