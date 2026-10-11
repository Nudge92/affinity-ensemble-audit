# 친화도 앙상블을 만들고, 넉 달 뒤 그 결론을 스스로 기각한 기록

> 2026년 4월에 8개 친화도 예측 모델을 6개 표적에서 묶어 **타깃별 앙상블이 단일 모델보다 낫다**는
> 결과를 냈습니다(평균 Spearman ρ 0.734 vs 0.607).
> 2026년 8월에 원자료를 다시 읽고 **그 향상치를 과적합 상한으로 판정했습니다.**
> 이 저장소는 앙상블이 아니라 **그 판정**이 내용입니다.

| | |
|---|---|
| 본작업 | 2026.03.23 ~ 04.20 (활동일 16일) |
| 자체감사 | 2026.08.18 · 08.24 (2일) |
| 감사 문서 | [`REVIEW.md`](REVIEW.md) — 원자료에서 직접 재계산 |
| 당시 README | [`README_original_2026-04-20.md`](README_original_2026-04-20.md) — **고치지 않고 그대로 둠** |

---

## 1. 무엇을 기각했나

### 1-1. 신호가 음성 대조군에서 왔다

CDK2 확장 세트(n=525)에서 8모델 전부를 두 조건으로 재계산했습니다.

| 모델 | ρ (전체 n≈523) | ρ (**실제 결합체만** n≈413) |
|---|---:|---:|
| DeepDTA | **0.276** | 0.149 |
| GIGN | 0.266 | 0.123 |
| PSICHIC | 0.226 | **0.012** |
| NMDN | 0.201 | 0.031 |
| GraphDTA | 0.086 | 0.045 |
| OnionNet2 | 0.073 | **−0.065** |
| MGraphDTA | −0.062 | 0.052 |
| PIGNet2 | −0.018 | 0.002 |

**음성 대조군을 빼면 8모델 전부가 −0.065 ~ +0.149 에 모입니다.**
전체 ρ 에 보이던 신호는 대부분 "결합하는 것과 결합하지 않는 것"을 가르는 성분이었고,
그건 친화도 모델 없이도 됩니다.

근거: [`results/06_validation/control_validation_cdk2_n500.csv`](results/06_validation/control_validation_cdk2_n500.csv) · [`REVIEW.md`](REVIEW.md) §1-3

### 1-2. 앙상블 향상치는 과적합 상한이다

타깃별 가중치를 **평가에 쓴 바로 그 데이터에서** 그리드서치(step 0.05)한 뒤
같은 데이터에서 ρ 를 보고했습니다. 홀드아웃도 nested CV 도 없었습니다.

| | 앙상블 평균 ρ | 최고 단일 모델 | 차이 |
|---|---:|---:|---:|
| 6타깃 평균 | 0.734 | 0.607 | **+0.127** |

→ 이 **+0.127 은 성능이 아니라 상한**입니다. `scripts/finalize_recipes.py` 가 그 그리드서치를 합니다.

근거: [`results/08_final_recipes/`](results/08_final_recipes/) · [`REVIEW.md`](REVIEW.md) §3-②

### 1-3. 기준선이 하나도 없었다

무작위 순위 · 분자량/logP 단독 · 암기(최근접 이웃 라벨 복사) · 도킹 점수 단독 · 상수 예측 —
**다섯 중 어느 것도 넣지 않았습니다.** 그래서 §1-1 의 "ρ ≈ 0" 이 *모델이 작동하지 않는다*인지
*평균을 찍고 있다*인지 당시에는 가를 수 없었습니다.

근거: [`REVIEW.md`](REVIEW.md) §3-①

---

## 2. 감사가 남긴 질문에 뒤늦게 답한 것

`REVIEW.md` §3-① 은 "PSICHIC 이 평균 근처를 찍고 있을 가능성"을 의심만 하고 닫지 못했습니다.
예측 분포를 보면 답이 나옵니다.

| 모델 | 실제 결합체 415종 예측 폭 | 예측 표준편차 |
|---|---|---:|
| **DeepDTA** | **[5.08, 6.07] = 0.99** | **0.191** |
| PSICHIC | [5.02, 7.98] = 2.96 | 0.494 |
| PIGNet2 | [1.40, 8.89] = 7.49 | 1.361 |

같은 415종의 **실험 pKd 는 폭 4.90, 표준편차 0.772** 입니다.
→ DeepDTA 의 출력 폭은 실험 폭의 **20%**, 표준편차는 **25%** 입니다.

**실제 결합체에서 ρ 1위(0.149)인 모델이 출력 범위가 가장 좁은 모델입니다.**
상수 기준선이 없으면 "작동한다"와 "거의 상수를 낸다"를 가를 수 없다는 §3-① 의 지적이
수치로 확인됩니다. 의심한 대상(PSICHIC)과 실제로 해당하는 모델(DeepDTA)은 달랐습니다.

---

## 3. ★ 감사 자체도 한 군데 과했다

`REVIEW.md` §3-③ 은 음성 대조군의 `Exp_pKd` 가 **"2.0 또는 3.0 으로 임의 배정"** 됐다고 적었습니다.
n=525 판본을 실제로 세어 보면 그렇지 않습니다.

| 음성 대조군 110건 | 건수 |
|---|---:|
| 손으로 배정한 2.0 / 3.0 | **10건** (Aspirin · Ibuprofen · Metformin 등) |
| ChEMBL IC50 → pKd 변환 | **100건** (`Notes` 열에 `ChEMBL:CHEMBL…; IC50 converted to pKd` 기록) |

**110건 중 10건만** 임의 배정이고, ChEMBL 확장이 그 문제를 대부분 이미 고쳐 둔 상태였습니다.
감사 문서는 초기 소규모 판본(n=25)의 상황을 확장 판본에까지 그대로 적용했습니다.

→ **§1-1 의 붕괴는 여전히 성립합니다**(임의 라벨 10건을 빼도 양성만 남기면 ρ 는 그대로 사라집니다).
다만 그 원인을 "임의 라벨 때문"으로 돌린 §3-③ 의 설명은 **과했습니다.**

`REVIEW.md` 는 고치지 않고 그대로 두었습니다. 이 절이 그 정정입니다.

---

## 4. 무엇을 만들었나 (본작업)

| 단계 | 내용 |
|---|---|
| 생성 | Pocket2Mol (E3-GNN) 로 포켓 기반 de novo 분자 생성 |
| 필터 | ADMET_AI · Admetica 로 물성 필터 |
| 복합체 | 3D 입력이 필요한 모델은 SMINA 도킹으로 포즈 생성 |
| **예측** | **8모델** — 서열기반 4(DeepDTA · GraphDTA · MGraphDTA · PSICHIC) + 구조기반 4(PIGNet2 · OnionNet2 · GIGN · NMDN) |
| 표적 | **6종** — CDK2 · EGFR · HIV-1 PR · PTP1B · ERα · CA-II |
| 앙상블 | 가중 평균, 타깃별 그리드서치(step 0.05) |
| 통계 | 부트스트랩 95% CI · 순열검정 · 스태킹 비교 |
| 파이프라인 | [`Snakefile`](Snakefile) · 타깃별 `prepare_*_benchmark.sh` |

★ **훈련/검증 분할이 없습니다.** 외부 사전학습 모델 8개를 대조군 세트에 그대로 추론한 구조입니다.

---

## 5. 이 저장소가 주장하지 않는 것

| 주장하지 않는 것 | 이유 |
|---|---|
| **"앙상블이 단일 모델보다 낫다"** | 가중치를 평가 데이터에서 찾았습니다(§1-2). +0.127 은 상한입니다 |
| **"어느 모델이 가장 낫다"** | 실제 결합체만 보면 8모델이 −0.065 ~ +0.149 에 모입니다(§1-1) |
| **데이터 누수를 점검했다** | 하지 않았습니다. 양성 대조군에 스타우로스포린·AT7519 같은 유명 저해제가 있고, 이들은 PDBbind·Davis 에 들어 있을 가능성이 높습니다 |
| **절대 친화도 예측이 쓸 만하다** | RMSE 가 2.2 ~ 5.6 pKd 입니다. 공개 벤치마크에서 잘 만든 스코어링 함수는 1.3 ~ 1.6 수준입니다 |
| **최신 co-folding 모델과 비교했다** | Boltz · Chai-1 · AlphaFold3 · NeuralPLexer 는 들어 있지 않습니다(작업 시점 문제) |

---

## 6. 데이터와 라이선스 — 경로별로 다릅니다

**코드는 MIT**([`LICENSE`](LICENSE)), **데이터는 아닙니다.** 전체 내역은 [`data/LICENSE-DATA`](data/LICENSE-DATA) 에 있습니다.

| 경로 | 출처 | 조건 |
|---|---|---|
| `src/` · `scripts/` · `Snakefile` · `prepare_*.sh` | 자체 작성 | **MIT** |
| `data/control_ligands_cdk2_expanded.csv`<br>`data/control_ligands_cdk2_new_only.csv`<br>`data/control_ligands_cdk2_n500.csv`<br>`data/control_ligands_cdk2_n500_new_only.csv` | **ChEMBL** (EMBL-EBI) | **CC BY-SA 3.0** |
| `results/06_validation/` · `results/09_dataset_audit/`<br>`results/10_baseline/` · `results/12_fep_pairs/` | 위 ChEMBL 데이터의 파생물 | **CC BY-SA 3.0** |
| `data/*.pdb` | **RCSB PDB** | 각 배포처 조건 |
| 그 밖의 `results/` | 이 저장소 코드의 산출물 | MIT (단 입력에 ChEMBL 이 섞였으면 위 조건이 함께 적용) |

ChEMBL: https://www.ebi.ac.uk/chembl/ · CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
RCSB PDB: https://www.rcsb.org/

이 저장소는 각 배포처 약관의 전문을 옮기지 않으며 해석도 하지 않습니다.
데이터를 쓰려면 위 링크에서 원문을 직접 확인해야 합니다.

**수집 스크립트**: `scripts/fetch_chembl_cdk2.py` · `scripts/fetch_chembl_cdk2_ic50_nothreshold.py` · `scripts/fetch_chembl_cdk2_n110_backup.py`

---

## 7. 이 저장소에 없는 것

- **`external_models/`** (8.7 GB) — 외부 모델 8종의 저장소 클론. 각 저장소에서 직접 받아야 합니다.
  (DeepDTA · GraphDTA · MGraphDTA · PSICHIC · PIGNet2 · OnionNet-2 · GIGN · DiffDock-NMDN)
- **`cache/`** (2.5 GB) — 중간 캐시
- **`logs/`** — 실행 로그. 사용자 홈 절대경로가 많아 제외했습니다
- 학습된 모델 가중치

**체크포인트 출처는 검증하지 않았습니다** — 각 모델이 실제로 어떤 가중치를 썼는지,
SMINA 포즈의 품질이 어땠는지는 확인하지 않은 채로 남아 있습니다([`REVIEW.md`](REVIEW.md) 부록).

---

## 8. 재현

실행 경로의 절대경로는 `conda run -n <env> python` 형태로 바꿔 두었습니다.
어느 환경이 필요한지는 그대로 읽힙니다 — `envs/` 에 정의가 있습니다.

```bash
# 외부 모델 8종을 external_models/ 아래에 각각 클론
# envs/*.yaml 로 환경 생성
snakemake -s Snakefile -j 1
```

★ 모델마다 환경이 다릅니다(`ADMET_AI` · `Admetica` · `Pocket2Mol` · `deepdta` · `nmdn` ·
`onionnet2` · `pignet2` · `psichic_fp` · `md`). 한 환경에서 전부 돌지 않습니다.
