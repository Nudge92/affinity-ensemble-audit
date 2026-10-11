#!/usr/bin/env bash
# ================================================================
# prepare_egfr_benchmark.sh
# EGFR 벤치마크 준비 스크립트
# 서버에서 실행: bash prepare_egfr_benchmark.sh
# ================================================================
set -e

PROJECT_DIR=~/Project/PLI
DATA_DIR=$PROJECT_DIR/data
SCRIPTS_DIR=$PROJECT_DIR/scripts

echo "=============================================="
echo "  EGFR Benchmark 준비"
echo "=============================================="

# ── 1. PDB 다운로드 (4WKQ: WT EGFR + gefitinib, 1.85Å) ──
echo ""
echo "[1/4] PDB 4WKQ 다운로드..."
cd $DATA_DIR
wget -q https://files.rcsb.org/download/4WKQ.pdb -O 4WKQ_raw.pdb
echo "  다운로드 완료: $(wc -l < 4WKQ_raw.pdb) lines"

# ── 2. PDB 전처리 (Chain A만, 리간드/물 제거) ──
echo ""
echo "[2/4] PDB 전처리..."
# Chain 확인
echo "  Chains: $(grep '^ATOM' 4WKQ_raw.pdb | awk '{print $5}' | sort -u | tr '\n' ' ')"
echo "  Ligands: $(grep '^HETATM' 4WKQ_raw.pdb | awk '{print $4}' | sort -u | tr '\n' ' ')"

# Chain A만 추출, HETATM(리간드/물) 제거
grep -E "^(ATOM)" 4WKQ_raw.pdb | grep " A " > egfr_clean.pdb
# END 추가
echo "END" >> egfr_clean.pdb
echo "  전처리 완료: egfr_clean.pdb ($(grep -c '^ATOM' egfr_clean.pdb) atoms)"

# ── 3. Pocket center 추출 (gefitinib 좌표 기준) ──
echo ""
echo "[3/4] Pocket center 추출 (gefitinib = 0WM in 4WKQ)..."
# gefitinib의 리간드 코드 확인 후 중심 좌표 계산
# 4WKQ에서 gefitinib 리간드 코드: 0WM (또는 다른 코드일 수 있음)
python3 << 'PYTHON'
import sys

# 4WKQ에서 HETATM 리간드 좌표 읽기
hetatm_lines = []
with open("4WKQ_raw.pdb") as f:
    for line in f:
        if line.startswith("HETATM") and "HOH" not in line:
            hetatm_lines.append(line)

if not hetatm_lines:
    print("  WARNING: No HETATM found, check PDB manually")
    sys.exit(0)

# 리간드 종류 출력
lig_names = set()
for line in hetatm_lines:
    lig_names.add(line[17:20].strip())
print(f"  Ligand residues found: {lig_names}")

# 첫 번째 비-물 리간드의 중심 좌표 계산
xs, ys, zs = [], [], []
target_lig = None
for line in hetatm_lines:
    resname = line[17:20].strip()
    if resname != "HOH":
        if target_lig is None:
            target_lig = resname
        if resname == target_lig:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xs.append(x)
                ys.append(y)
                zs.append(z)
            except:
                pass

if xs:
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    print(f"  Ligand: {target_lig}")
    print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
    print(f"  N atoms: {len(xs)}")
    
    # pocket center를 파일에 저장
    with open("egfr_pocket_center.txt", "w") as f:
        f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
    print(f"  Saved to egfr_pocket_center.txt")
else:
    print("  ERROR: Could not extract ligand coordinates")
PYTHON

# ── 4. 대조군 CSV 복사 ──
echo ""
echo "[4/4] 대조군 CSV 확인..."
if [ -f "$DATA_DIR/control_ligands_egfr.csv" ]; then
    echo "  control_ligands_egfr.csv already exists"
else
    echo "  ERROR: control_ligands_egfr.csv not found in $DATA_DIR"
    echo "  Please copy it manually"
fi

N_POS=$(grep -c "positive" $DATA_DIR/control_ligands_egfr.csv 2>/dev/null || echo 0)
N_NEG=$(grep -c "negative" $DATA_DIR/control_ligands_egfr.csv 2>/dev/null || echo 0)
echo "  양성 대조군: ${N_POS}개"
echo "  음성 대조군: ${N_NEG}개"

# ── 요약 ──
echo ""
echo "=============================================="
echo "  준비 완료!"
echo "=============================================="
echo "  PDB: $DATA_DIR/egfr_clean.pdb"
echo "  Pocket: $DATA_DIR/egfr_pocket_center.txt"
echo "  Controls: $DATA_DIR/control_ligands_egfr.csv"
echo ""
echo "  다음 단계:"
echo "    1. pocket center 확인 후 validate_with_controls_egfr.py 수정"
echo "    2. python scripts/validate_with_controls_egfr.py 실행"
echo "=============================================="
