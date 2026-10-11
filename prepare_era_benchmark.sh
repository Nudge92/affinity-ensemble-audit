#!/usr/bin/env bash
# ================================================================
# prepare_era_benchmark.sh
# ERα (Estrogen Receptor Alpha) LBD 벤치마크 준비
# PDB: 3ERT (ERα LBD + 4-hydroxytamoxifen, 1.9Å, monomer Chain A)
# ================================================================
set -e

PROJECT_DIR=~/Project/PLI
DATA_DIR=$PROJECT_DIR/data

echo "=============================================="
echo "  ERα (Estrogen Receptor Alpha) Benchmark 준비"
echo "=============================================="

# ── 1. PDB 다운로드 ──
echo ""
echo "[1/4] PDB 3ERT 다운로드..."
cd $DATA_DIR
wget -q https://files.rcsb.org/download/3ERT.pdb -O 3ERT_raw.pdb
echo "  다운로드 완료: $(wc -l < 3ERT_raw.pdb) lines"
echo "  Chains: $(grep '^ATOM' 3ERT_raw.pdb | awk '{print $5}' | sort -u | tr '\n' ' ')"
echo "  Ligands: $(grep '^HETATM' 3ERT_raw.pdb | awk '{print $4}' | sort -u | tr '\n' ' ')"

# ── 2. PDB 전처리 (Chain A만) ──
echo ""
echo "[2/4] PDB 전처리 (Chain A, HETATM 제거)..."
grep "^ATOM" 3ERT_raw.pdb | grep " A " > era_clean.pdb
echo "END" >> era_clean.pdb
echo "  전처리 완료: era_clean.pdb ($(grep -c '^ATOM' era_clean.pdb) atoms)"

# ── 3. Pocket center 추출 (OHT = 4-hydroxytamoxifen) ──
echo ""
echo "[3/4] Pocket center 추출..."
python3 << 'PYTHON'
lines = open("3ERT_raw.pdb").readlines()
skip_ligs = {"HOH","NA","CL","MES","GOL","EDO","SO4","PO4","ACT","BME","PEG","DMS","MPD"}

lig_names = set()
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        if rn not in skip_ligs:
            lig_names.add(rn)
print(f"  Drug-like ligands: {lig_names}")

# OHT (4-hydroxytamoxifen) = Chain A의 리간드
xs, ys, zs = [], [], []
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        chain = line[21:22].strip()
        if rn not in skip_ligs and chain == "A":
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except:
                pass

if xs:
    cx = sum(xs)/len(xs)
    cy = sum(ys)/len(ys)
    cz = sum(zs)/len(zs)
    print(f"  Chain A ligand ({len(xs)} atoms)")
    print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
    with open("era_pocket_center.txt", "w") as f:
        f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
else:
    # fallback: all chains
    for line in lines:
        if line.startswith("HETATM"):
            rn = line[17:20].strip()
            if rn not in skip_ligs:
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except:
                    pass
    if xs:
        cx = sum(xs)/len(xs)
        cy = sum(ys)/len(ys)
        cz = sum(zs)/len(zs)
        print(f"  All chains ligand ({len(xs)} atoms)")
        print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
        with open("era_pocket_center.txt", "w") as f:
            f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
    else:
        print("  WARNING: No ligand found")
PYTHON

# ── 4. 대조군 확인 ──
echo ""
echo "[4/4] 대조군 CSV 확인..."
N_POS=$(grep -c "positive" $DATA_DIR/control_ligands_era.csv 2>/dev/null || echo 0)
N_NEG=$(grep -c "negative" $DATA_DIR/control_ligands_era.csv 2>/dev/null || echo 0)
echo "  양성 대조군: ${N_POS}개"
echo "  음성 대조군: ${N_NEG}개"

POCKET=$(cat $DATA_DIR/era_pocket_center.txt 2>/dev/null || echo "N/A")
echo ""
echo "=============================================="
echo "  준비 완료!"
echo "=============================================="
echo "  PDB: $DATA_DIR/era_clean.pdb"
echo "  Pocket: $POCKET"
echo "  Controls: $DATA_DIR/control_ligands_era.csv"
echo ""
echo "  ERα 특성:"
echo "    - Monomer LBD, ~250 aa"
echo "    - Nuclear receptor (NOT kinase/protease)"
echo "    - Davis: 없음 → Davis 모델 OOD"
echo "    - PDBbind: 풍부"
echo "    - 양성 대조군: nM급(estradiol~tamoxifen) ~ µM급(genistein, BPA)"
echo "=============================================="