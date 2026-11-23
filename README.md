# SepsisGuard — 패혈증 골든타임 방어를 위한 합동 심문 멀티 에이전트 시스템

ICU 환자의 바이탈 시계열과 EMR(검사·배양·영상 판독)을 각각 다른 에이전트가 분석하고, 두 에이전트의 판단을 **합동 심문(attention 융합)** 으로 교차 검증해 **현재 시점 기준 4시간 이내 패혈증 발병 위험**을 예측합니다. 책임 의사 모듈이 결과를 Sepsis-3 관련 지표와 대조해 의료진용 리포트를 만듭니다.

- 🔗 **시연 대시보드 (가상 환자)**: https://doovinkim.github.io/sepsisguard_demo/
- 📁 시연 페이지 저장소: https://github.com/DooVinKim/sepsisguard_demo

> 연구용 코드입니다. 진단·처방을 대체하지 않습니다.
> 이 저장소에는 **MIMIC-IV 데이터, 파생 데이터, 학습된 모델 가중치가 없습니다** (PhysioNet 데이터 사용 약정).

---

## 팀

| 이름 | GitHub |
|---|---|
| 김두빈 | [@DooVinKim](https://github.com/DooVinKim) |
| 노민 |  |
| 남건우 |  |

---

## 시스템 구조

```
 바이탈 24h (HR·SBP·DBP·MAP·RR·SpO2·Temp·Shock Index) + 나이·성별
        │
        ▼
 ┌──────────────────┐      context 128d
 │ 바이탈 에이전트   │ ───────────────┐
 │ 2층 LSTM + attn  │                │
 └──────────────────┘                ▼
                              ┌──────────────┐      ┌────────────────────┐
                              │  합동 심문     │ ───▶ │ 책임 의사 모듈       │ ──▶ 위험 등급 + 리포트
                              │ 3-토큰 attn   │      │ Sepsis-3 지표 대조   │
                              └──────────────┘      └────────────────────┘
 ┌──────────────────┐                ▲
 │ EMR 에이전트      │ ───────────────┘
 │ PubMedBERT 미세조정│      문서 벡터 768d
 └──────────────────┘
        ▲
 검사(48h) · 배양 채취/결과 · 영상 판독 (T 이전 정보만) → 시점별 "EMR 스냅샷 문서"로 직렬화
```

- **바이탈 에이전트**: 결측은 보간하지 않고 마스크로 처리, 지수 감쇠 가중치(최근 측정 중시), 2층 LSTM(hidden 128) + 시점 attention. attention 가중치는 "어느 시점이 결정적이었나" 설명에 사용.
- **EMR 에이전트**: 수치·범주·텍스트를 자연어 문서 하나로 직렬화한 뒤 PubMedBERT(`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, PubMed 사전학습, MIMIC 미포함)를 전체 미세조정.
- **합동 심문**: `[CLS, vital, emr]` 3토큰 self-attention. CLS attention = 각 에이전트 근거 의존도(XAI).
- **책임 의사 모듈**: 규칙 기반. 젖산·혈소판·빌리루빈·크레아티닌·백혈구 플래그, qSOFA 2항목(호흡수·수축기 혈압, 의식 상태 제외), 점수 경과, 영상 판독 키워드(부정·호전 문맥 제외)로 리포트 문장 생성.

## 라벨링 (시점 단위)

환자 단위 0/1 이 아니라 **시간 단위 시점 T 마다** 라벨을 붙입니다.

- MIT-LCP Sepsis-3 정의로 환자별 발병 시점 추출
- 발병 **0–4시간 전** 시점 = 1, 그 외(비패혈증 환자 전체, 패혈증 환자의 더 이른 시점) = 0, 발병 이후 시점은 제외
- 모든 입력은 **T 이전에 기록·결과가 나온 정보만** 사용 (배양 결과·영상 판독도 결과 시각 기준) → 시간 누출 방지
- train / val / test 는 **환자 단위** 분할 (같은 환자가 두 집합에 섞이지 않음)

## 결과 (융합 test 70,476 시점, 양성 4.76%)

| 구성 | AUROC |
|---|---|
| 바이탈 에이전트 단독 | 0.726 |
| EMR 에이전트 단독 | 0.752 |
| 단순 결합 (concat + 로지스틱) | 0.802 |
| **합동 심문 (attention 융합)** | **0.815** ± 0.0001 (3 seed), AUPRC 0.237 |

각 에이전트 단독 성능의 한계를 융합으로 넘었습니다. 점수는 확률 보정 전 상대 위험 점수입니다.

---

## 저장소 구성

```
preprocess/   R + DuckDB 전처리
  00_sepsis3_generation.R    MIT-LCP concepts_duckdb SQL 실행 → sepsis3 발병 시점
  00_config.R                경로 · 피처 itemid · 생리학적 범위 (다른 스크립트가 source)
  01b_chartevents_extract.R  chartevents 에서 ICU 바이탈 추출 (lazy 스캔)
  02c_vital_label_split.R    시점 라벨링 · 이상치 정제 · 리샘플링 · 정규화 · 환자 단위 split
  03_emr_timepoints.R        바이탈과 같은 (stay, T) 시점 그리드
  04c_emr_windowing.R        검사 · 배양 · 영상 윈도잉 (T 이전 가용 정보만)
  05c_emr_serialize.R        시점별 EMR 스냅샷 문서 직렬화
vital/        바이탈 에이전트 학습(vital_v3.py) · 모델 정의/로더(vital_v3_model.py)
emr/          EMR 에이전트 학습 (train_emr.py — 인코더/LLM, LoRA/전체 미세조정 공용)
fusion/       합동 심문: 정렬 데이터 생성 → EMR 벡터 교체 → 융합 학습
system/       통합 엔진 + 책임 의사 리포트 + 가상 환자 · 시연 데이터 생성
```

## 실행 순서

MIMIC-IV 접근 권한(PhysioNet 자격 인증 + DUA)이 있어야 재현할 수 있습니다. 원본은 `data/` 에 두고, 경로는 `preprocess/00_config.R` 와 각 스크립트 인자로 맞춥니다.

```bash
# 1) 전처리 (preprocess/ 폴더에서 실행)
Rscript 00_sepsis3_generation.R          # MIT-LCP mimic-code 의 concepts_duckdb 필요
Rscript 01b_chartevents_extract.R
Rscript 02c_vital_label_split.R 0 4 2 3 b0_4_relax
Rscript 03_emr_timepoints.R b0_4_relax
Rscript 04c_emr_windowing.R b0_4_relax 48
Rscript 05c_emr_serialize.R b0_4_relax 48

# 2) 바이탈 에이전트 (저장소 루트에서)
python vital/vital_v3.py --tag b0_4_relax --window 24 --run_name T4_relax_w24_all

# 3) EMR 에이전트 (전체 미세조정, 양성 가중치 5) → 융합용 벡터 내보내기
M=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
python emr/train_emr.py --docs <EMR 문서 parquet> --model $M --run pubmedbert_full_fullft_pw5 --full_ft --lr 1e-5 --pos_weight 5 --eval_every <N>
python emr/train_emr.py --docs <EMR 문서 parquet> --model $M --run pubmedbert_full_fullft_pw5 --full_ft --export_from models/pubmedbert_full_fullft_pw5

# 4) 합동 심문
python fusion/fusion_06a_v3.py                                   # 바이탈 context + 그리드 정렬
python fusion/fusion_06a_swap_emr.py --emr preds/pubmedbert_full_fullft_pw5 --out data/fusion_data_pubmedbert_pw5
python fusion/fusion_06b_v4.py --data data/fusion_data_pubmedbert_pw5 --name pubmedbert_pw5 --seed 1   # seed 42, 2 도

# 5) 통합 엔진 · 가상 환자 시연 데이터
python system/make_synthetic_patients.py
python system/run_engine.py            # 학습된 가중치 필요
python system/build_data_js.py --out <시연 저장소>/data
```

## 환경

- 전처리: R (data.table, arrow, duckdb)
- 학습: Python, PyTorch, transformers, peft, scikit-learn, pandas, pyarrow

## 데이터와 가중치에 대해

- 데이터: [MIMIC-IV](https://physionet.org/content/mimiciv/) (hosp · icu 모듈), [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) (radiology만 사용; 퇴원요약은 시간 누출 때문에 제외)
- Sepsis-3 정의: [MIT-LCP mimic-code](https://github.com/MIT-LCP/mimic-code) `concepts_duckdb`
- MIMIC-IV 로 학습한 모델 가중치는 데이터 사용 약정상 공개하지 않습니다.
- `system/patients/` 의 환자는 전부 **가상 환자**이며 실제 환자 데이터가 아닙니다.
