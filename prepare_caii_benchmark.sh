#!/usr/bin/env bash
# ================================================================
# prepare_caii_benchmark.sh  
# CA-II (Carbonic Anhydrase II) 벤치마크 준비
# PDB: 3HS4 (CA-II + acetazolamide/AZM, 1.1Å, monomer)
# ================================================================
set -e

PROJECT_DIR=~/Project/PLI
DATA_DIR=$PROJECT_DIR/data

echo "=============================================="
echo "  CA-II (Carbonic Anhydrase II) Benchmark 준비"
echo "=============================================="

echo ""
echo "[1/4] PDB 3HS4 다운로드..."
cd $DATA_DIR
wget -q https://files.rcsb.org/download/3HS4.pdb -O 3HS4_raw.pdb
echo "  다운로드 완료: $(wc -l < 3HS4_raw.pdb) lines"
echo "  Chains: $(grep '^ATOM' 3HS4_raw.pdb | awk '{print $5}' | sort -u | tr '\n' ' ')"
echo "  Ligands: $(grep '^HETATM' 3HS4_raw.pdb | awk '{print $4}' | sort -u | tr '\n' ' ')"

echo ""
echo "[2/4] PDB 전처리 (Chain A, HETATM 제거)..."
grep "^ATOM" 3HS4_raw.pdb | grep " A " > caii_clean.pdb
echo "END" >> caii_clean.pdb
echo "  전처리 완료: caii_clean.pdb ($(grep -c '^ATOM' caii_clean.pdb) atoms)"

echo ""
echo "[3/4] Pocket center 추출 (AZM = acetazolamide)..."
python3 << 'PYTHON'
lines = open("3HS4_raw.pdb").readlines()
skip_ligs = {"HOH","NA","CL","MES","GOL","EDO","SO4","PO4","ACT","BME","PEG","DMS","MPD","ZN","GOL"}

lig_names = set()
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        if rn not in skip_ligs:
            lig_names.add(rn)
print(f"  Ligand residues found: {lig_names}")

# AZM (acetazolamide) Chain A
xs, ys, zs = [], [], []
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        chain = line[21:22].strip()
        if rn == "AZM" and chain == "A":
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except:
                pass

# If no Chain A AZM, try any AZM
if not xs:
    for line in lines:
        if line.startswith("HETATM"):
            rn = line[17:20].strip()
            if rn == "AZM":
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
    print(f"  Ligand: AZM ({len(xs)} atoms)")
    print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
    with open("caii_pocket_center.txt", "w") as f:
        f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
    print(f"  Saved to caii_pocket_center.txt")
else:
    print("  WARNING: AZM not found, trying first non-solvent ligand")
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
        print(f"  Fallback pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
        with open("caii_pocket_center.txt", "w") as f:
            f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
PYTHON

echo ""
echo "[4/4] 대조군 CSV 확인..."
N_POS=$(grep -c "positive" $DATA_DIR/control_ligands_caii.csv 2>/dev/null || echo 0)
N_NEG=$(grep -c "negative" $DATA_DIR/control_ligands_caii.csv 2>/dev/null || echo 0)
echo "  양성 대조군: ${N_POS}개"
echo "  음성 대조군: ${N_NEG}개"

POCKET=$(cat $DATA_DIR/caii_pocket_center.txt 2>/dev/null || echo "N/A")
echo ""
echo "=============================================="
echo "  준비 완료!"
echo "=============================================="
echo "  PDB: $DATA_DIR/caii_clean.pdb"
echo "  Pocket: $POCKET"
echo "  Controls: $DATA_DIR/control_ligands_caii.csv"
echo ""
echo "  CA-II 특성:"
echo "    - Monomer, ~260 aa"
echo "    - Zinc metalloenzyme"  
echo "    - Davis: 없음 → Davis 모델 OOD"
echo "    - PDBbind: 매우 풍부 (sulfonamide co-crystal 다수)"
echo "    - 양성: nM급 sulfonamide 위주 (acetazolamide~dorzolamide)"
echo "=============================================="