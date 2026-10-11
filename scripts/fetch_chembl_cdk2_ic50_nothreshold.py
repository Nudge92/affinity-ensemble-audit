#!/usr/bin/env python3
"""
fetch_chembl_cdk2.py — ChEMBL에서 CDK2 활성 데이터 수집 및 정제
================================================================
ChEMBL에서 CDK2 결합 데이터를 가져와서:
  1) IC50, Ki, Kd → pKd로 통일
  2) 품질 필터링 (assay confidence, 단위, 이상값)
  3) 양성(pKd ≥ 6.5) / 음성(pKd ≤ 4.0) 분류
  4) 기존 25개와 중복 제거
  5) 양성 60개 + 음성 25개 = 총 85개 선별
  6) 기존 control_ligands_cdk2.csv 포맷으로 저장

사용법:
  cd ~/Project/PLI
  python scripts/fetch_chembl_cdk2.py

출력:
  data/control_ligands_cdk2_expanded.csv  (총 ~85개)
  data/control_ligands_cdk2_new_only.csv  (기존 제외, 신규만)
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from chembl_webresource_client.new_client import new_client

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.expanduser("~/Project/PLI")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXISTING_CSV = os.path.join(DATA_DIR, "control_ligands_cdk2.csv")
OUTPUT_EXPANDED = os.path.join(DATA_DIR, "control_ligands_cdk2_n500.csv")
OUTPUT_NEW_ONLY = os.path.join(DATA_DIR, "control_ligands_cdk2_n500_new_only.csv")

# CDK2 ChEMBL ID
CDK2_CHEMBL_ID = "CHEMBL301"

# pKd 기준
POS_THRESHOLD = None   # 임계 필터 제거 (연속 분포)    # pKd ≥ 6.5 → positive (Kd ≤ 316 nM)
NEG_THRESHOLD = None   # 임계 필터 제거 (연속 분포)    # pKd ≤ 4.0 → negative (Kd ≥ 100 µM)

# 샘플링 목표
N_POSITIVE = 400
N_NEGATIVE = 100


# ═══════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════

def ic50_to_pkd(value_nm: float) -> float:
    """IC50 (nM) → pKd 근사 변환. -log10(value / 1e9)"""
    if value_nm <= 0:
        return np.nan
    return -math.log10(value_nm * 1e-9)


def standard_value_to_pkd(row) -> float:
    """
    ChEMBL의 standard_value + standard_units + standard_type → pKd 변환
    지원 타입: IC50, Ki, Kd, EC50 (참고용)
    지원 단위: nM, µM, mM
    """
    stype = str(row.get("standard_type", "")).upper()
    value = row.get("standard_value")
    units = str(row.get("standard_units", "")).strip()

    if value is None or pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan

    if value <= 0:
        return np.nan

    # 단위 → nM 통일
    if units == "nM":
        value_nm = value
    elif units in ["µM", "uM", "μM"]:
        value_nm = value * 1000
    elif units == "mM":
        value_nm = value * 1_000_000
    elif units == "pM":
        value_nm = value / 1000
    else:
        return np.nan  # 알 수 없는 단위

    # IC50/Ki/Kd → pKd
    if stype in ["IC50", "KI", "KD", "EC50"]:
        return ic50_to_pkd(value_nm)
    else:
        return np.nan


def is_valid_smiles(smi: str) -> bool:
    """RDKit으로 SMILES 유효성 검사"""
    if not smi or pd.isna(smi):
        return False
    mol = Chem.MolFromSmiles(str(smi))
    return mol is not None


def canonical_smiles(smi: str) -> str:
    """정규화된 SMILES 반환"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def lipinski_ok(smi: str) -> bool:
    """기본 Lipinski Rule of Five 필터"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    # 약간 완화된 기준 (천연물, 매크로사이클 포함)
    return mw <= 900 and logp <= 8 and hbd <= 10 and hba <= 15


# ═══════════════════════════════════════════════════════════════
# Step 1: ChEMBL에서 데이터 다운로드
# ═══════════════════════════════════════════════════════════════

def fetch_chembl_data(target_id: str) -> pd.DataFrame:
    """ChEMBL에서 타겟의 활성 데이터 다운로드"""
    print(f"\n{'='*60}")
    print(f"  Step 1: ChEMBL에서 {target_id} 데이터 다운로드")
    print(f"{'='*60}")

    activity = new_client.activity
    results = activity.filter(
        target_chembl_id=target_id,
        standard_type__in=["IC50"],
        standard_relation__in=["=", "<"],   # 정확한 값 또는 < (상한값)
        assay_type="B",                      # Binding assay만
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "assay_chembl_id",
        "assay_description",
        "pchembl_value",                     # ChEMBL이 자체 계산한 pActivity
        "activity_comment",
        "data_validity_comment",
    ])

    print(f"  다운로드 중... (인터넷 연결 필요, 30초~2분 소요)")
    data = list(results)
    df = pd.DataFrame(data)
    print(f"  원본 레코드 수: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════
# Step 2: 데이터 정제
# ═══════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ChEMBL 원본 데이터 정제"""
    print(f"\n{'='*60}")
    print(f"  Step 2: 데이터 정제")
    print(f"{'='*60}")
    n_start = len(df)

    # 2-1. SMILES 없는 것 제거
    df = df.dropna(subset=["canonical_smiles"])
    print(f"  SMILES 없는 것 제거: {n_start} → {len(df)}")

    # 2-2. 데이터 유효성 문제 있는 것 제거
    invalid_comments = ["Outside typical range", "Non standard unit",
                        "Potential transcription error", "Potential missing data"]
    if "data_validity_comment" in df.columns:
        mask = df["data_validity_comment"].isna() | \
               (~df["data_validity_comment"].isin(invalid_comments))
        df = df[mask]
        print(f"  유효성 문제 제거 후: {len(df)}")

    # 2-3. pKd 계산
    # ChEMBL 자체 계산값(pchembl_value)이 있으면 우선 사용, 없으면 직접 계산
    def get_pkd(row):
        if row.get("pchembl_value") and not pd.isna(row.get("pchembl_value")):
            try:
                return float(row["pchembl_value"])
            except:
                pass
        return standard_value_to_pkd(row)

    df["pKd"] = df.apply(get_pkd, axis=1)
    df = df.dropna(subset=["pKd"])
    print(f"  pKd 계산 후: {len(df)}")

    # 2-4. pKd 범위 필터 (3.0 ~ 11.0)
    df = df[(df["pKd"] >= 3.0) & (df["pKd"] <= 11.0)]
    print(f"  pKd 범위 필터(3~11) 후: {len(df)}")

    # 2-5. SMILES 유효성 검사
    df = df[df["canonical_smiles"].apply(is_valid_smiles)]
    print(f"  SMILES 유효성 검사 후: {len(df)}")

    # 2-6. Canonical SMILES 정규화
    df["canonical_smiles"] = df["canonical_smiles"].apply(canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    # 2-7. 약물성 필터 (Lipinski 완화 기준)
    df = df[df["canonical_smiles"].apply(lipinski_ok)]
    print(f"  약물성 필터 후: {len(df)}")

    # 2-8. 같은 분자 중복 → pKd 중앙값으로 통합
    df_dedup = df.groupby("canonical_smiles").agg({
        "pKd": "median",
        "molecule_chembl_id": "first",
        "standard_type": "first",
    }).reset_index()
    print(f"  중복 제거 후: {len(df_dedup)}")

    return df_dedup


# ═══════════════════════════════════════════════════════════════
# Step 3: 양성/음성 분류 및 샘플링
# ═══════════════════════════════════════════════════════════════

def classify_and_sample(df: pd.DataFrame, existing_smiles: set) -> pd.DataFrame:
    """양성/음성 분류, 기존 분자 제외, 샘플링"""
    print(f"\n{'='*60}")
    print(f"  Step 3: 분류 및 샘플링")
    print(f"{'='*60}")

    # 기존 분자 제외
    df = df[~df["canonical_smiles"].isin(existing_smiles)].copy()
    print(f"  기존 분자 제외 후: {len(df)}")

    # 양성/음성 분류
    # 임계 필터 제거: 전량 연속 사용
    pos = df.copy()
    neg = df.iloc[0:0].copy()

    print(f"  양성(pKd≥{POS_THRESHOLD}): {len(pos)}개")
    print(f"  음성(pKd≤{NEG_THRESHOLD}): {len(neg)}개")

    # 양성 샘플링: pKd 분포가 골고루 되도록 (높은 것만 치우치지 않게)
    if len(pos) >= N_POSITIVE:
        # pKd 구간별로 균등 샘플링
        pos["pkd_bin"] = pd.cut(pos["pKd"], bins=5, labels=False)
        n_per_bin = N_POSITIVE // 5
        sampled_pos = []
        for b in range(5):
            bin_df = pos[pos["pkd_bin"] == b]
            n = min(n_per_bin, len(bin_df))
            sampled_pos.append(bin_df.sample(n=n, random_state=42))
        # 나머지 채우기
        sampled_pos_df = pd.concat(sampled_pos)
        if len(sampled_pos_df) < N_POSITIVE:
            remaining = pos[~pos.index.isin(sampled_pos_df.index)]
            extra = remaining.sample(
                n=min(N_POSITIVE - len(sampled_pos_df), len(remaining)),
                random_state=42)
            sampled_pos_df = pd.concat([sampled_pos_df, extra])
        pos_final = sampled_pos_df.head(N_POSITIVE)
    else:
        pos_final = pos
        print(f"  ⚠️ 양성 {N_POSITIVE}개 목표이지만 {len(pos)}개만 확보됨")

    # 음성 샘플링
    if len(neg) >= N_NEGATIVE:
        neg_final = neg.sample(n=N_NEGATIVE, random_state=42)
    else:
        neg_final = neg
        print(f"  ⚠️ 음성 {N_NEGATIVE}개 목표이지만 {len(neg)}개만 확보됨")

    print(f"\n  최종 선별:")
    print(f"    양성: {len(pos_final)}개 (pKd {pos_final['pKd'].min():.2f} ~ {pos_final['pKd'].max():.2f})")
    print(f"    음성: {len(neg_final)}개 (pKd {neg_final['pKd'].min():.2f} ~ {neg_final['pKd'].max():.2f})")

    # 합치기
    pos_final["Type"] = "positive"
    neg_final["Type"] = "negative"
    combined = pd.concat([pos_final, neg_final], ignore_index=True)

    return combined


# ═══════════════════════════════════════════════════════════════
# Step 4: CSV 포맷으로 저장
# ═══════════════════════════════════════════════════════════════

def format_and_save(new_df: pd.DataFrame, existing_df: pd.DataFrame):
    """기존 CSV 포맷으로 변환 후 저장"""
    print(f"\n{'='*60}")
    print(f"  Step 4: 저장")
    print(f"{'='*60}")

    # 기존 포맷: ID, Drug_Name, SMILES, Type, pKd, Notes
    n_existing = len(existing_df)

    records = []
    pos_count = sum(existing_df["Type"] == "positive")
    neg_count = sum(existing_df["Type"] == "negative")

    for _, row in new_df.iterrows():
        is_pos = row["Type"] == "positive"
        if is_pos:
            pos_count += 1
            idx = pos_count
            prefix = "CDK2_pos"
        else:
            neg_count += 1
            idx = neg_count
            prefix = "CDK2_neg"

        chembl_id = row.get("molecule_chembl_id", "unknown")
        assay_type = row.get("standard_type", "IC50")

        records.append({
            "ID": f"{prefix}_{idx:02d}",
            "Drug_Name": chembl_id,
            "SMILES": row["canonical_smiles"],
            "Type": row["Type"],
            "pKd": round(row["pKd"], 2),
            "Notes": f"ChEMBL:{chembl_id}; {assay_type} converted to pKd",
        })

    new_formatted = pd.DataFrame(records)

    # 신규 데이터만 저장
    new_formatted.to_csv(OUTPUT_NEW_ONLY, index=False)
    print(f"  신규 데이터 저장: {OUTPUT_NEW_ONLY} ({len(new_formatted)}개)")

    # 기존 + 신규 합친 버전 저장
    expanded = pd.concat([existing_df, new_formatted], ignore_index=True)

    # ID 재부여
    pos_rows = expanded[expanded["Type"] == "positive"].reset_index(drop=True)
    neg_rows = expanded[expanded["Type"] == "negative"].reset_index(drop=True)
    pos_rows["ID"] = [f"CDK2_pos_{i+1:02d}" for i in range(len(pos_rows))]
    neg_rows["ID"] = [f"CDK2_neg_{i+1:02d}" for i in range(len(neg_rows))]
    expanded = pd.concat([pos_rows, neg_rows], ignore_index=True)

    expanded.to_csv(OUTPUT_EXPANDED, index=False)
    print(f"  확장 데이터 저장: {OUTPUT_EXPANDED}")
    print(f"    기존 {n_existing}개 + 신규 {len(new_formatted)}개 = 총 {len(expanded)}개")
    print(f"    양성: {sum(expanded['Type']=='positive')}개")
    print(f"    음성: {sum(expanded['Type']=='negative')}개")

    return expanded


# ═══════════════════════════════════════════════════════════════
# Step 5: 요약 통계
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  최종 요약")
    print(f"{'='*60}")

    pos = df[df["Type"] == "positive"]
    neg = df[df["Type"] == "negative"]

    print(f"\n  전체: {len(df)}개")
    print(f"  양성({len(pos)}개):")
    print(f"    pKd 평균: {pos['pKd'].mean():.2f} ± {pos['pKd'].std():.2f}")
    print(f"    pKd 범위: {pos['pKd'].min():.2f} ~ {pos['pKd'].max():.2f}")
    print(f"  음성({len(neg)}개):")
    print(f"    pKd 평균: {neg['pKd'].mean():.2f} ± {neg['pKd'].std():.2f}")
    print(f"    pKd 범위: {neg['pKd'].min():.2f} ~ {neg['pKd'].max():.2f}")
    print(f"\n  출력 파일:")
    print(f"    {OUTPUT_EXPANDED}")
    print(f"    {OUTPUT_NEW_ONLY}")
    print(f"\n  다음 단계:")
    print(f"    python scripts/validate_with_controls_cdk2.py \\")
    print(f"      --input data/control_ligands_cdk2_expanded.csv")


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CDK2 Control Ligands 확장 (ChEMBL)")
    print("=" * 60)

    # 기존 데이터 로드
    existing_df = pd.read_csv(EXISTING_CSV)
    existing_smiles = set()
    for smi in existing_df["SMILES"]:
        canon = canonical_smiles(smi)
        if canon:
            existing_smiles.add(canon)
    print(f"\n  기존 데이터: {len(existing_df)}개 ({len(existing_smiles)}개 unique SMILES)")

    # Step 1: 다운로드
    raw_df = fetch_chembl_data(CDK2_CHEMBL_ID)

    # Step 2: 정제
    clean_df = clean_data(raw_df)

    # Step 3: 분류 + 샘플링
    sampled_df = classify_and_sample(clean_df, existing_smiles)

    if len(sampled_df) == 0:
        print("\n⚠️ 유효한 데이터가 없습니다. 네트워크 연결을 확인하세요.")
        return

    # Step 4: 저장
    final_df = format_and_save(sampled_df, existing_df)

    # Step 5: 요약
    print_summary(final_df)


if __name__ == "__main__":
    main()
