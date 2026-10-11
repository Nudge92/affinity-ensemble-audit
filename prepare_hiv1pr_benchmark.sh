#!/usr/bin/env bash
# ================================================================
# prepare_hiv1pr_benchmark.sh
# HIV-1 Protease 벤치마크 준비 + 실행 스크립트
# 서버에서: bash prepare_hiv1pr_benchmark.sh
# ================================================================
set -e

PROJECT_DIR=~/Project/PLI
DATA_DIR=$PROJECT_DIR/data

echo "=============================================="
echo "  HIV-1 Protease Benchmark 준비"
echo "=============================================="

# ── 1. PDB 다운로드 (3NU3: WT HIV-1 PR + amprenavir, 1.35Å) ──
echo ""
echo "[1/5] PDB 3NU3 다운로드..."
cd $DATA_DIR
wget -q https://files.rcsb.org/download/3NU3.pdb -O 3NU3_raw.pdb
echo "  다운로드 완료: $(wc -l < 3NU3_raw.pdb) lines"

# 구조 확인
echo "  Chains: $(grep '^ATOM' 3NU3_raw.pdb | awk '{print $5}' | sort -u | tr '\n' ' ')"
echo "  Ligands: $(grep '^HETATM' 3NU3_raw.pdb | awk '{print $4}' | sort -u | tr '\n' ' ')"
echo "  Resolution: $(grep 'RESOLUTION' 3NU3_raw.pdb | head -1)"

# ── 2. PDB 전처리 ──
# HIV-1 protease는 homodimer (Chain A + B, 각 99 aa)
# 양쪽 chain 모두 필요 (활성 부위가 dimer interface에 있음)
echo ""
echo "[2/5] PDB 전처리 (Chain A+B, 리간드/물 제거)..."
grep "^ATOM" 3NU3_raw.pdb > hiv1pr_clean.pdb
echo "END" >> hiv1pr_clean.pdb
echo "  전처리 완료: hiv1pr_clean.pdb ($(grep -c '^ATOM' hiv1pr_clean.pdb) atoms)"

# ── 3. Pocket center 추출 ──
echo ""
echo "[3/5] Pocket center 추출..."
python3 << 'PYTHON'
lines = open("3NU3_raw.pdb").readlines()
# 리간드 종류 확인
lig_names = set()
for line in lines:
    if line.startswith("HETATM") and "HOH" not in line:
        lig_names.add(line[17:20].strip())
print(f"  Ligand residues: {lig_names}")

# amprenavir 리간드 코드 찾기 (478 = amprenavir in PDB)
xs, ys, zs = [], [], []
for line in lines:
    if line.startswith("HETATM") and "HOH" not in line:
        resname = line[17:20].strip()
        if resname not in ("HOH", "NA", "CL", "MES", "GOL", "EDO", "SO4", "PO4", "ACT", "BME"):
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
    print(f"  Pocket center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
    print(f"  N ligand atoms: {len(xs)}")
    with open("hiv1pr_pocket_center.txt", "w") as f:
        f.write(f"{cx:.1f},{cy:.1f},{cz:.1f}\n")
    print(f"  Saved to hiv1pr_pocket_center.txt")
else:
    print("  WARNING: No ligand atoms found")
PYTHON

# ── 4. HIV-1 protease 서열 (WT, HXB2 strain, 99 aa per monomer) ──
echo ""
echo "[4/5] HIV-1 protease 서열 확인..."
# HIV-1 protease monomer sequence (99 aa, HXB2/NL4-3 reference)
HIV1PR_SEQ="PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
echo "  HIV-1 protease monomer: ${#HIV1PR_SEQ} aa"
echo "  Sequence: ${HIV1PR_SEQ:0:50}..."
echo "  NOTE: Davis 데이터셋에 HIV-1 protease 없음 → DeepDTA/GraphDTA/MGraphDTA는 out-of-distribution"

# ── 5. 대조군 CSV 확인 ──
echo ""
echo "[5/5] 대조군 CSV 확인..."
N_POS=$(grep -c "positive" $DATA_DIR/control_ligands_hiv1pr.csv 2>/dev/null || echo 0)
N_NEG=$(grep -c "negative" $DATA_DIR/control_ligands_hiv1pr.csv 2>/dev/null || echo 0)
echo "  양성 대조군: ${N_POS}개"
echo "  음성 대조군: ${N_NEG}개"

# ── 6. Pocket center 읽기 ──
POCKET=$(cat $DATA_DIR/hiv1pr_pocket_center.txt 2>/dev/null || echo "0,0,0")
echo ""
echo "=============================================="
echo "  준비 완료!"
echo "=============================================="
echo "  PDB: $DATA_DIR/hiv1pr_clean.pdb"
echo "  Pocket: $POCKET"
echo "  Controls: $DATA_DIR/control_ligands_hiv1pr.csv"
echo "  서열: 99 aa (monomer), NOT in Davis dataset"
echo ""
echo "  다음: validate_with_controls_hiv1pr.py의 POCKET_CENTER 수정 후 실행"
echo "=============================================="
