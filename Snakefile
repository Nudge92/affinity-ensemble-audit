# =============================================================================
# Snakefile — PLI 앙상블 파이프라인 (Phase 4)
# =============================================================================
# Phase 3 확정 레시피(ensemble_recipes.json)를 적용하여
# 가중 앙상블 파이프라인을 자동화합니다.
#
# 현재: CDK2 단일 타겟으로 end-to-end 검증
# 확장: run_affinity.py를 타겟별 파라미터 대응으로 수정 후 6타겟 확장 가능
#
# 실행:
#   snakemake --cores 1 -n              # dry-run
#   snakemake --cores 1                 # 실행 (GPU 1개이므로 cores=1 권장)
#   snakemake --cores 1 -R ensemble_affinity  # 앙상블만 재실행
#
# 각 모델은 별도 conda 환경에 설치되어 있으며,
# run_*.py 내부에서 subprocess로 해당 환경을 호출합니다.
# =============================================================================
import json
import os

# ── 레시피 로드 ──
RECIPE_PATH = "results/08_final_recipes/ensemble_recipes.json"
if os.path.exists(RECIPE_PATH):
    with open(RECIPE_PATH) as f:
        RECIPES = json.load(f)
else:
    RECIPES = None

# ── 타겟 단백질 설정 (현재 CDK2만 활성) ──
PDB_PATH = "data/cdk2_clean.pdb"
POCKET_CENTER = "0.5,27.1,9.0"
TARGET = "cdk2"

# ── 모델 리스트 ──
GEN_MODELS   = ["Pocket2Mol"]
ADMET_MODELS = ["ADMET_AI", "Admetica"]

# 레시피 기반 affinity 모델 결정
if RECIPES and TARGET in RECIPES.get("target_specific", {}):
    _recipe = RECIPES["target_specific"][TARGET]
    RECIPE_MODELS  = _recipe["models"]
    RECIPE_WEIGHTS = dict(zip(_recipe["models"], _recipe["weights"]))
    EXCLUDED       = _recipe.get("excluded", [])
else:
    # fallback: 전체 모델 동일 가중치
    RECIPE_MODELS  = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA",
                      "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]
    RECIPE_WEIGHTS = {m: 1.0/8 for m in RECIPE_MODELS}
    EXCLUDED       = []

# 앙상블에 사용할 모델 (레시피 모델만)
AFFINITY_MODELS = RECIPE_MODELS

# 전체 모델 (affinity 룰에서 사용 — 기존 결과 파일이 있는 모델 포함)
ALL_AFFINITY_MODELS = [
    "PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA",
    "MGraphDTA", "GIGN", "PSICHIC", "NMDN",
]


# ── 최종 목표 ──
rule all:
    input:
        expand("results/04_ensemble/{gen}-{admet}-weighted_ensemble.csv",
               gen=GEN_MODELS, admet=ADMET_MODELS),


# ── 1. 분자 생성 ──
rule generate:
    output:
        "results/01_gen/{gen}-molecules.csv"
    log:
        "logs/01_gen/{gen}.log"
    shell:
        """
        cp results/01_gen/Pocket2Mol-CDK2-molecules.csv {output} 2>/dev/null \
            || (echo "ERROR: Pocket2Mol-CDK2-molecules.csv 없음" && exit 1)
        echo "CDK2 분자 CSV 연결 완료: $(wc -l < {output}) lines" > {log} 2>&1
        """


# ── 2. ADMET 필터링 ──
rule admet:
    input:
        "results/01_gen/{gen}-molecules.csv"
    output:
        "results/02_admet/{gen}-{admet}-filtered.csv"
    log:
        "logs/02_admet/{gen}-{admet}.log"
    params:
        model_name="{admet}"
    script:
        "scripts/run_admet.py"


# ── 3. Affinity 예측 (기존 스크립트 호환) ──
rule affinity:
    input:
        ligands="results/02_admet/{gen}-{admet}-filtered.csv",
        protein=PDB_PATH,
    output:
        "results/03_affinity/{gen}-{admet}-{aff}-scores.csv"
    log:
        "logs/03_affinity/{gen}-{admet}-{aff}.log"
    params:
        model_name="{aff}",
    script:
        "scripts/run_affinity.py"


# ── 4. 가중 앙상블 (Phase 3 레시피 적용) ──
rule ensemble_affinity:
    input:
        scores=expand(
            "results/03_affinity/{{gen}}-{{admet}}-{aff}-scores.csv",
            aff=AFFINITY_MODELS),
    output:
        "results/04_ensemble/{gen}-{admet}-weighted_ensemble.csv"
    log:
        "logs/04_ensemble/{gen}-{admet}-weighted_ensemble.log"
    params:
        weights_json=json.dumps(RECIPE_WEIGHTS),
        target_name=TARGET,
        family=RECIPES["target_specific"][TARGET]["family"] if RECIPES else "kinase",
        excluded=json.dumps(EXCLUDED),
    script:
        "scripts/run_ensemble_weighted.py"


# ── 5. (참고) 기존 동일가중 앙상블 — 비교용 ──
rule ensemble_equal:
    input:
        scores=expand(
            "results/03_affinity/{{gen}}-{{admet}}-{aff}-scores.csv",
            aff=ALL_AFFINITY_MODELS),
    output:
        "results/04_ensemble/{gen}-{admet}-equal_ensemble.csv"
    log:
        "logs/04_ensemble/{gen}-{admet}-equal_ensemble.log"
    script:
        "scripts/run_ensemble.py"