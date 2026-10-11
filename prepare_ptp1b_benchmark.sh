#!/usr/bin/env bash
# ================================================================
# prepare_ptp1b_benchmark.sh
# PTP1B (Protein Tyrosine Phosphatase 1B) 벤치마크 준비
# PDB: 2QBP (PTP1B + inhibitor, 1.55Å, monomer Chain A)
# ================================================================
set -e

PROJECT_DIR=~/Project/PLI
DATA_DIR=$PROJECT_DIR/data

echo "=============================================="
echo "  PTP1B Benchmark 준비"
echo "=============================================="

# ── 1. PDB 다운로드 ──
echo ""
echo "[1/4] PDB 2QBP 다운로드..."
cd $DATA_DIR
wget -q https://files.rcsb.org/download/2QBP.pdb -O 2QBP_raw.pdb
echo "  다운로드 완료: $(wc -l < 2QBP_raw.pdb) lines"
echo "  Chains: $(grep '^ATOM' 2QBP_raw.pdb | awk '{print $5}' | sort -u | tr '\n' ' ')"
echo "  Ligands: $(grep '^HETATM' 2QBP_raw.pdb | awk '{print $4}' | sort -u | tr '\n' ' ')"

# ── 2. PDB 전처리 (Monomer, Chain A만) ──
echo ""
echo "[2/4] PDB 전처리 (Chain A, HETATM 제거)..."
grep "^ATOM" 2QBP_raw.pdb | grep " A " > ptp1b_clean.pdb
echo "END" >> ptp1b_clean.pdb
echo "  전처리 완료: ptp1b_clean.pdb ($(grep -c '^ATOM' ptp1b_clean.pdb) atoms)"

# ── 3. Pocket center 추출 ──
echo ""
echo "[3/4] Pocket center 추출..."
python3 << 'PYTHON'
lines = open("2QBP_raw.pdb").readlines()

# 리간드 종류 확인
lig_names = set()
skip_ligs = {"HOH", "NA", "CL", "MES", "GOL", "EDO", "SO4", "PO4", "ACT", "BME", "PEG", "DMS", "MPD"}
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        if rn not in skip_ligs:
            lig_names.add(rn)
print(f"  Drug-like ligands: {lig_names}")

# 첫 번째 drug-like 리간드의 중심 좌표
xs, ys, zs = [], [], []
target_lig = None
for line in lines:
    if line.startswith("HETATM"):
        rn = line[17:20].strip()
        if rn not in skip_ligs:
            if target_lig is None:
                target_lig = rn
            if rn == target_lig:
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
    print(f"  Ligand: {target_lig} ({len(xs)} atoms)")
    print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
    with open("ptp1b_pocket_center.txt", "w") as f:
        f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
else:
    print("  WARNING: No ligand found")
PYTHON

# ── 4. 대조군 확인 ──
echo ""
echo "[4/4] 대조군 CSV 확인..."
N_POS=$(grep -c "positive" $DATA_DIR/control_ligands_ptp1b.csv 2>/dev/null || echo 0)
N_NEG=$(grep -c "negative" $DATA_DIR/control_ligands_ptp1b.csv 2>/dev/null || echo 0)
echo "  양성 대조군: ${N_POS}개"
echo "  음성 대조군: ${N_NEG}개"

POCKET=$(cat $DATA_DIR/ptp1b_pocket_center.txt 2>/dev/null || echo "N/A")
echo ""
echo "=============================================="
echo "  준비 완료!"
echo "=============================================="
echo "  PDB: $DATA_DIR/ptp1b_clean.pdb"
echo "  Pocket: $POCKET"
echo "  Controls: $DATA_DIR/control_ligands_ptp1b.csv"
echo ""
echo "  PTP1B 특성:"
echo "    - Monomer, ~300 aa (Chain A)"
echo "    - Phosphatase (NOT kinase)"
echo "    - Davis: 없음 → DeepDTA/GraphDTA/MGraphDTA OOD"
echo "    - PDBbind: 풍부"
echo "    - PSICHIC: 300 aa → OOM 없음"
echo "    - PIGNet2: monomer → graph 생성 가능"
echo ""
echo "  다음: pocket center로 validate 스크립트 수정 후 실행"
echo "=============================================="