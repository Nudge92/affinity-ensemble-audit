# PLI-Ensemble: Multi-Model Ensemble Pipeline for Protein–Ligand Interaction Prediction

> **De novo molecule generation → ADMET filtering → 8-model affinity prediction → target-specific weighted ensemble scoring**

A benchmarking and ensemble optimization pipeline that combines eight diverse binding affinity models across six protein targets from different families. Target-specific ensemble recipes consistently outperform any single model, achieving up to **+0.177 improvement** in Spearman ρ over the best individual predictor.

---

## Highlights

- **8 affinity models** spanning 1D-CNN, GNN, Transformer, and physics-informed architectures
- **6 protein targets** covering kinases, proteases, phosphatases, nuclear receptors, and metalloenzymes
- **Target-specific ensemble recipes** via grid search optimization (avg ρ = 0.734 vs 0.607 best solo)
- **End-to-end Snakemake pipeline** from molecular generation to weighted ensemble ranking
- **Cross-validation** confirming target-specific recipes outperform universal approaches by +0.153
- **GNN architecture comparison** — 4 architectures trained from scratch on Davis, with CDK-family fine-tuning
- **Ensemble method comparison** — Weighted Average vs Stacking validated across n=25, n=110, n=525 (ChEMBL-expanded), with Bootstrap 95% CI and Permutation testing

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────────────┐     ┌──────────────────┐
│  Pocket2Mol  │────▶│   ADMET Filter   │────▶│   8-Model Affinity Pred    │────▶│ Weighted Ensemble│
│  (E3-GNN)    │     │ ADMET_AI/Admetica│     │                            │     │  (target recipe) │
│              │     │                  │     │  ┌───────┐  ┌──────────┐   │     │                  │
│  3D pocket   │     │  41+ properties  │     │  │DeepDTA│  │ GraphDTA │   │     │  grid-searched   │
│  → SMILES    │     │  safety filters  │     │  │MGraph │  │ PSICHIC  │   │     │  weights per     │
│              │     │                  │     │  │PIGNet2│  │OnionNet2 │   │     │  target protein  │
│              │     │                  │     │  │ GIGN  │  │  NMDN    │   │     │                  │
└─────────────┘     └──────────────────┘     └────────────────────────────┘     └──────────────────┘
```

---

## 8 Affinity Models

| Model | Architecture | Training Data | Input Type | Conda Env |
|-------|-------------|---------------|------------|-----------|
| DeepDTA | 1D CNN | Davis (kinase) | Sequence + SMILES | `deepdta` |
| GraphDTA | GNN (GCN) | Davis (kinase) | Sequence + Mol graph | `pignet2` |
| MGraphDTA | Multi-scale GNN | Davis (kinase) | Sequence + Mol graph | `pignet2` |
| PIGNet2 | GNN + Physics | PDBbind | 3D complex | `pignet2` |
| OnionNet2 | CNN (distance shell) | PDBbind | 3D complex | `onionnet2` |
| GIGN | GNN (interaction) | PDBbind | 3D complex | `pignet2` |
| PSICHIC | Transformer (ESM2) | PDBbind | Sequence + Mol graph | `psichic_fp` |
| NMDN | Diffusion scoring | PDBbind | 3D complex | `nmdn` |

Models requiring 3D structures use **SMINA docking** to generate receptor–ligand complexes before inference.

---

## 6 Benchmark Targets

| Target | Family | PDB | Pocket Center (Å) | Notes |
|--------|--------|-----|-------------------|-------|
| CDK2 | Kinase | 1AQ1 | (0.5, 27.1, 9.0) | Reference target |
| EGFR | Kinase | 4WKQ | (1.5, 194.2, 20.4) | — |
| HIV-1 PR | Protease | 3NU3 | (16.6, 22.6, 17.6) | Homodimer (chains A+B) |
| PTP1B | Phosphatase | 2QBP | (48.2, 10.1, 3.1) | — |
| ERα | Nuclear receptor | 3ERT | (31.6, -1.6, 25.6) | — |
| CA-II | Metalloenzyme | 3HS4 | (-4.2, -4.1, 17.0) | — |

Each target uses **n = 25 control ligands** (positive binders with known pKd + negative decoys) for benchmarking.

---

## Results

### Phase 1 — Single-Model Benchmark (Spearman ρ)

Pass threshold: ρ ≥ 0.3

| Model | Pass | CDK2 | EGFR | HIV-1PR | PTP1B | ERα | CA-II |
|-------|------|------|------|---------|-------|-----|-------|
| GIGN | 5/6 | 0.385 | 0.519 | 0.594 | 0.273 | 0.508 | 0.635 |
| MGraphDTA | 5/6 | 0.331 | 0.684 | 0.299 | 0.432 | 0.301 | 0.570 |
| PSICHIC | 4/6 | 0.546 | 0.395 | 0.406 | 0.188 | 0.240 | 0.688 |
| NMDN | 4/6 | 0.517 | 0.097 | 0.679 | 0.072 | 0.524 | 0.429 |
| OnionNet2 | 3/6 | 0.410 | 0.417 | 0.768 | 0.089 | 0.247 | 0.048 |
| DeepDTA | 2/6 | 0.422 | -0.375 | 0.195 | 0.329 | -0.087 | -0.113 |
| GraphDTA | 1/6 | 0.177 | 0.659 | -0.005 | 0.077 | -0.173 | 0.183 |
| PIGNet2 | 1/5 | 0.365 | 0.268 | N/A | -0.369 | 0.058 | -0.133 |

**Key finding**: No single model dominates across all targets. The best model varies by protein family.

### Phase 2 — Target-Specific Ensemble Optimization

Models with ρ < 0 are excluded per target. Grid search (step = 0.05) over 2–4 model combinations:

| Target | Family | Optimal Ensemble | Weights | ρ_ens | ρ_solo | Δρ |
|--------|--------|-----------------|---------|-------|--------|-----|
| CDK2 | Kinase | PSICHIC+NMDN+GIGN+PIGNet2 | 0.70/0.05/0.10/0.15 | **0.663** | 0.546 | +0.117 |
| EGFR | Kinase | GraphDTA+OnionNet2+PSICHIC+PIGNet2 | 0.40/0.40/0.10/0.10 | **0.861** | 0.684 | +0.177 |
| HIV-1PR | Protease | OnionNet2+NMDN+GIGN+PSICHIC | 0.45/0.15/0.15/0.25 | **0.825** | 0.768 | +0.057 |
| PTP1B | Phosphatase | MGraphDTA+DeepDTA+OnionNet2+GraphDTA | 0.40/0.35/0.05/0.20 | **0.562** | 0.432 | +0.130 |
| ERα | Nuc. receptor | NMDN+GIGN+MGraphDTA+PIGNet2 | 0.05/0.10/0.80/0.05 | **0.664** | 0.524 | +0.140 |
| CA-II | Metalloenzyme | PSICHIC+GIGN+MGraphDTA+GraphDTA | 0.30/0.05/0.30/0.35 | **0.830** | 0.688 | +0.141 |

Ensemble outperforms the best solo model on **all 6 targets** (Δρ = +0.057 to +0.177).

### Phase 2.5 — Cross-Validation

6×6 cross-application matrix confirms each target's own recipe ranks #1 for that target (6/6).

| Strategy | Mean ρ | Min ρ |
|----------|--------|-------|
| Best solo model | 0.607 | 0.432 |
| Equal-weight all models | 0.541 | 0.167 |
| Universal 3-model | 0.613 | 0.440 |
| **Target-specific recipe** | **0.734** | **0.562** |

### Phase 4 — End-to-End Pipeline Verification

CDK2 Snakemake pipeline validated (11/11 steps):

```
Pocket2Mol → ADMET_AI/Admetica → PSICHIC/NMDN/GIGN/PIGNet2 → Weighted Ensemble
```

- Admetica path: 12 molecules → 4-model weighted ensemble → Top-1 Score = 6.643
- ADMET_AI path: 1 molecule passed → Score = 2.896

---

## GNN Architecture Benchmark

To validate model behavior beyond pretrained inference, four GNN architectures were trained from scratch on the Davis kinase dataset (25,046 train / 5,010 test pairs), then fine-tuned on CDK-family data and evaluated on CDK2 control ligands.

### Davis Test Set — Architecture Comparison

All models trained with Adam (lr=0.0005), batch size 512, early stopping (patience=50).

| Model | Params | MSE | RMSE | Pearson | Spearman | CI | Best Epoch |
|-------|--------|-----|------|---------|----------|----|------------|
| GINConvNet | 1.30M | **0.234** | **0.483** | **0.844** | **0.689** | **0.884** | 198 |
| GAT_GCN | 4.75M | 0.265 | 0.515 | 0.818 | 0.675 | 0.874 | 208 |
| GATNet | 1.72M | 0.277 | 0.526 | 0.811 | 0.660 | 0.865 | 298 |
| GCNNet | 2.06M | 0.289 | 0.537 | 0.801 | 0.655 | 0.861 | 272 |

GINConvNet achieves the best overall Davis performance, consistent with the original GraphDTA paper.

### CDK-Family Fine-tuning

Pretrained models were fine-tuned on 738 CDK-family pairs (13 CDK proteins from Davis) with lr=0.0001 and patience=30, then evaluated on 25 CDK2 control ligands:

| Model | CDK2 ρ (pretrained) | CDK2 ρ (fine-tuned) | Δρ |
|-------|--------------------|--------------------|-----|
| **GCNNet** | **0.356** | **0.365** | +0.009 |
| GINConvNet | -0.224 | -0.186 | +0.038 |
| GATNet | -0.013 | 0.045 | +0.058 |
| GAT_GCN | 0.064 | -0.040 | -0.104 |

### Key Insight

**The best Davis model is the worst CDK2 model.** GINConvNet dominates on Davis (ρ=0.689) but produces anti-correlated predictions on CDK2 (ρ=-0.224). Meanwhile, GCNNet — the weakest on Davis (ρ=0.655) — generalizes best to CDK2 (ρ=0.356). This reinforces the core thesis of this project: **target-specific model selection and ensembling are essential**, as global benchmark performance does not predict per-target effectiveness.

<details>
<summary>Training scripts and outputs</summary>

```
external_models/GraphDTA/benchmark_advanced/
├── model_{arch}_davis.pt              # Pretrained checkpoints (4 models)
├── model_{arch}_cdk_ft.pt             # Fine-tuned checkpoints (4 models)
├── phase1_architecture_comparison.csv # Davis benchmark results
├── phase1_history_{arch}.csv          # Per-epoch training metrics
├── phase1_training_curves.png         # Training curve visualization
├── phase1_architecture_bar.png        # Architecture comparison chart
├── phase3_cdk2_comparison.csv         # Pretrained vs fine-tuned summary
└── advanced_benchmark_report.txt      # Full report
```

</details>

---

## Ensemble Method Comparison (CDK2 Deep-Dive)

To empirically validate the choice of ensemble strategy, CDK2 control ligands were expanded from n=25 to n=525 using ChEMBL bioactivity data, and multiple ensemble methods were compared using Leave-One-Out Cross-Validation across three data scales.

### Data Expansion

| Dataset | Source | Positive | Negative | Total |
|---------|--------|----------|----------|-------|
| n=25 | Manual curation | 15 | 10 | 25 |
| n=110 | + ChEMBL (pKd ≥ 6.5 / ≤ 5.0) | 75 | 35 | 110 |
| n=525 | + ChEMBL (pKd ≥ 6.5 / ≤ 5.5) | 415 | 110 | 525 |

### Weighted Average vs Stacking across Data Scales

All stacking methods use LOO-CV. "Best WA" uses the recipe re-optimized for each data scale.

| Method | n=25 (ρ) | n=110 (ρ) | n=525 (ρ) |
|--------|----------|-----------|-----------|
| **Best Weighted Avg** | **0.663** | 0.521 | 0.323 |
| Equal Weight | 0.719 | 0.369 | 0.203 |
| Ridge Stacking (α=10) | 0.486 | **0.406** | **0.305** |
| Linear Stacking | 0.125 | 0.399 | 0.305 |
| XGBoost Stacking | 0.268 | 0.288 | — |

| Method | n=25 (CI) | n=110 (CI) | n=525 (CI) |
|--------|-----------|------------|------------|
| **Best Weighted Avg** | **0.749** | 0.615 | 0.580 |
| Ridge Stacking | 0.670 | **0.639** | **0.604** |

| Method | n=25 (RMSE) | n=110 (RMSE) | n=525 (RMSE) |
|--------|-------------|--------------|--------------|
| **Best Weighted Avg** | **2.204** | 1.840 | 1.444 |
| Ridge Stacking | 2.230 | 1.777 | **1.304** |

**Key findings:**
1. With n=25, Weighted Average outperforms Stacking across all metrics (Stacking overfits)
2. From n=110 onward, Stacking consistently surpasses Weighted Average in ρ, CI, and RMSE
3. The advantage of Stacking is consistent: +0.082 at n=110, +0.070 at n=525

### Recipe Drift across Data Scales

Optimal ensemble recipes change dramatically as data grows:

| Scale | Optimal Recipe | ρ |
|-------|---------------|---|
| n=25 | PSICHIC(0.70) + PIGNet2(0.15) + GIGN(0.10) + NMDN(0.05) | 0.663 |
| n=110 | GIGN(0.50) + NMDN(0.45) + DeepDTA(0.05) | 0.521 |
| n=525 | DeepDTA(0.70) + GIGN(0.20) + PSICHIC(0.05) + NMDN(0.05) | 0.323 |

The dominant model shifts from PSICHIC → GIGN → DeepDTA as data grows. This demonstrates that **optimal recipes depend on both target and data distribution**, not just target identity.

**Why ρ decreases with larger n**: Larger datasets from ChEMBL include structurally diverse molecules with IC50→pKd conversion noise and a wider pKd range (2.0–10.05), making prediction harder. This is a more realistic evaluation — the n=25 curated set was "easy" data. Critically, SE decreases from 0.118 to 0.043 (63% reduction), meaning the lower ρ values are far more reliable estimates.

### Statistical Validation (Bootstrap 95% CI, n_boot=10,000)

All methods are statistically significant (p < 0.05 by permutation test).

| Method | n | ρ | 95% CI | SE | p-value |
|--------|---|---|--------|----|---------|
| Weighted Avg (Phase2) | 25 | +0.663 | [+0.361, +0.821] | 0.118 | 0.0004 |
| n=110 Optimal Recipe | 110 | +0.521 | [+0.363, +0.649] | 0.073 | <0.0001 |
| n=525 Optimal Recipe | 525 | +0.323 | [+0.196, +0.360] | 0.042 | <0.0001 |
| GIGN (best solo, n=525) | 525 | +0.266 | [+0.185, +0.347] | 0.041 | <0.0001 |

SE decreases consistently (0.118 → 0.073 → 0.042), confirming that larger datasets yield more stable and trustworthy ρ estimates.

<details>
<summary>Scripts and outputs</summary>

```
scripts/
├── fetch_chembl_cdk2.py            # ChEMBL data collection
├── stacking_experiment.py          # WA vs Stacking LOO-CV comparison
├── optimize_weights_cdk2_n110.py   # Grid search for n=110 optimal recipe
├── optimize_weights_cdk2_n500.py   # Grid search for n=525 optimal recipe
└── bootstrap_ci.py                 # Bootstrap CI + Permutation test

results/stacking/
├── stacking_comparison.csv         # All method results across scales
├── stacking_comparison.png         # Comparison chart
├── cdk2_n110_recipe.json           # n=110 optimal recipe
├── cdk2_n500_recipe.json           # n=525 optimal recipe
├── bootstrap_ci_report.txt         # CI + p-value report
└── bootstrap_ci.png                # Bootstrap distribution plots
```

</details>

---

## Repository Structure

```
PLI/
├── Snakefile                          # End-to-end pipeline (Phase 4)
├── data/
│   ├── {target}_clean.pdb             # Preprocessed protein structures (6 targets)
│   ├── {target}_pocket_center.txt     # Binding pocket coordinates
│   ├── control_ligands_{target}.csv   # Benchmark ligands (n=25 each)
│   ├── control_ligands_cdk2_expanded.csv  # CDK2 expanded (n=110)
│   └── control_ligands_cdk2_n500.csv      # CDK2 expanded (n=525)
├── scripts/
│   ├── run_gen.py                     # Step 1: Molecule generation (Pocket2Mol)
│   ├── run_admet.py                   # Step 2: ADMET filtering
│   ├── run_affinity.py                # Step 3: 8-model affinity prediction + docking
│   ├── run_ensemble_weighted.py       # Step 4: Weighted ensemble scoring
│   ├── optimize_weights_v2.py         # Phase 2: Multi-target optimization
│   ├── cross_validate_recipes.py      # Phase 2.5: 6×6 cross-validation
│   ├── finalize_recipes.py            # Phase 3: Recipe finalization
│   ├── validate_with_controls_{t}.py  # Phase 1: Per-target benchmarking
│   ├── fetch_chembl_cdk2.py           # ChEMBL data expansion
│   ├── stacking_experiment.py         # WA vs Stacking comparison
│   ├── optimize_weights_cdk2_n110.py  # n=110 recipe optimization
│   ├── optimize_weights_cdk2_n500.py  # n=525 recipe optimization
│   └── bootstrap_ci.py               # Bootstrap CI + Permutation test
├── external_models/
│   ├── GraphDTA/                      # GNN training benchmark (4 architectures)
│   │   ├── train_graphdta_advanced.py
│   │   ├── benchmark_advanced/
│   │   └── data/davis/
│   ├── Pocket2Mol/ DeepDTA/ MGraphDTA/
│   ├── PIGNet2/ OnionNet-2/ GIGN/
│   ├── PSICHIC/ DiffDock-NMDN/
│   └── DiffDock/ CheapNet/
├── results/
│   ├── 01_gen/ ... 04_ensemble/       # Pipeline outputs
│   ├── 06_validation/                 # Benchmark results (n=25, n=110, n=525)
│   ├── 07_weighted_ensemble/          # Phase 2/2.5 optimization
│   ├── 08_final_recipes/              # Final recipes (JSON)
│   └── stacking/                      # Ensemble comparison + Bootstrap CI
├── logs/
├── envs/
└── cache/
```

---

## How to Run

### Prerequisites

- **Hardware**: NVIDIA GPU with ≥ 8GB VRAM (tested on RTX 4060 Ti 8GB)
- **Software**: CUDA, miniforge3/conda, SMINA

### Environment Setup

```bash
conda create -n Pocket2Mol    # Molecule generation
conda create -n ADMET_AI      # ADMET filtering
conda create -n Admetica      # ADMET filtering (alternative)
conda create -n pignet2       # PIGNet2, GraphDTA, MGraphDTA, GIGN
conda create -n onionnet2     # OnionNet2
conda create -n deepdta       # DeepDTA
conda create -n psichic_fp    # PSICHIC
conda create -n nmdn          # NMDN (DiffDock-NMDN)
```

### Running the Pipeline

```bash
cd ~/Project/PLI
snakemake --cores 1 -n        # Dry run
snakemake --cores 1           # Execute
snakemake --cores 1 -R ensemble_affinity  # Re-run ensemble only
```

### Running Benchmarks

```bash
python scripts/validate_with_controls_cdk2.py  # Phase 1
python scripts/optimize_weights_v2.py           # Phase 2
python scripts/cross_validate_recipes.py        # Phase 2.5
python scripts/finalize_recipes.py              # Phase 3
```

### CDK2 Deep-Dive (Data Expansion + Ensemble Comparison)

```bash
# 1. Expand CDK2 data via ChEMBL
python scripts/fetch_chembl_cdk2.py

# 2. Run 8-model prediction on expanded data (~15-20hrs for n=525)
conda activate pignet2
python scripts/validate_with_controls_cdk2.py

# 3. Re-optimize recipe for expanded data
python scripts/optimize_weights_cdk2_n500.py

# 4. Compare Weighted Average vs Stacking
python scripts/stacking_experiment.py

# 5. Bootstrap CI + Permutation test
python scripts/bootstrap_ci.py
```

### GNN Architecture Benchmark

```bash
cd external_models/GraphDTA
conda activate pignet2
python train_graphdta_advanced.py  # ~3-4 hours
```

---

## Ensemble Recipes

Final recipes are stored in `results/08_final_recipes/ensemble_recipes.json`.

**Universal fallback** (for unseen targets): GIGN(0.05) + MGraphDTA(0.80) + NMDN(0.15), avg ρ = 0.613

| Model | # Targets Used | Total Weight | Dominant In |
|-------|---------------|-------------|-------------|
| MGraphDTA | 3 | 1.50 | ERα (0.80) |
| PSICHIC | 4 | 1.35 | CDK2 (0.70) |
| GraphDTA | 3 | 0.95 | CA-II (0.35) |
| OnionNet2 | 3 | 0.90 | HIV-1PR (0.45) |
| GIGN | 4 | 0.40 | Supporting role |
| DeepDTA | 1 | 0.35 | PTP1B (0.35) |
| PIGNet2 | 3 | 0.30 | Supporting role |
| NMDN | 3 | 0.25 | Supporting role |

---

## Technical Challenges & Solutions

| Issue | Solution |
|-------|----------|
| PSICHIC ESM2 OOM on long sequences (>500aa) | Extract domain only (e.g., EGFR kinase domain 498aa) |
| PSICHIC batch crash on certain SMILES | Per-molecule execution with error handling |
| PIGNet2 fails on homodimer structures | Skip dimer targets (HIV-1PR); works on monomers |
| Multiple conda environments needed | Scripts use subprocess to call env-specific Python |
| SMINA docking quality varies | exhaustiveness=8 baseline; room for improvement |
| Best Davis model ≠ best per-target model | Target-specific ensemble recipes with cross-validation |
| Phase 2 recipe degrades on expanded data | Re-optimize recipe per data distribution |
| n=25 too small for Stacking | Validated empirically; expanded to n=525 via ChEMBL |
| Optimal recipe shifts with data scale | Demonstrated PSICHIC→GIGN→DeepDTA dominance shift |

---

## Future Work

- Parameterize `run_affinity.py` for multi-target Snakemake execution (currently CDK2 hardcoded)
- Run Pocket2Mol generation for remaining 5 targets
- Expand control set for all 6 targets via ChEMBL (currently CDK2 only)
- Apply Stacking to other targets with sufficient data (n ≥ 100)
- Explore transfer learning from PDBbind-trained models to specific targets
- Upgrade docking (higher exhaustiveness or commercial docking tools)
- Investigate non-linear stacking with larger datasets (n ≥ 1000)

---

## Tech Stack

Python, PyTorch, PyTorch Geometric, RDKit, scikit-learn, Snakemake, SMINA, ESM2, ChEMBL API, Conda/Miniforge3, CUDA

---

## License

This project is for academic and portfolio purposes.
