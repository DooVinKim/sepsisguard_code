# 가상 환자 (시연용 합성 데이터)

- **실제 환자 아님.** 모든 값은 시연용으로 지어낸 값이며, MIMIC 데이터에서 복사한 행·문장이 없다.
- 생성: `../make_synthetic_patients.py` (seed 2026, 다시 실행하면 같은 결과) · 점검: `../check_synthetic.py`
- 형식: `DASHBOARD_RECOMMENDATION.md` 4절 입력 형식 v2. 시간은 **ICU 입실 후 경과 시간(시간 단위)**, 실제 날짜 없음.
- 검사 10종의 이름·단위·정상범위, 배양 검체·검사명·균명 표기는 학습 문서와 같은 표기(집계 최빈값)를 따름.
- 각 시점 T 에서 엔진이 쓰는 범위 (학습과 동일): 바이탈 직전 24h · 검사 채혈 48h 안 + 결과 저장 ≤ T · 배양 채취 24h / 결과 72h (결과 저장 ≤ T 이면 결과, 아니면 pending) · 영상 촬영 72h 안 + 판독 저장 ≤ T

| ID | 시나리오 | 의도한 결과 (모델로 확인·조정 필요) |
|---|---|---|
| DEMO-01 | 폐렴 진행 | 점수가 시간에 따라 저 → 중 → 고위험 |
| DEMO-02 | 요로감염, 40h 에 대장균 결과 도착 | 중간 → 고위험 (결과 도착 전후 변화) |
| DEMO-03 | 바이탈만 나쁨 (수술 후) | 바이탈 높음 · EMR 낮음 → 충돌 |
| DEMO-04 | 검사만 나쁨 | EMR 이 융합 점수를 끌어올림 |
| DEMO-05 | 안정 | 저위험 유지 |
| DEMO-06 | 치료 후 호전 | 고 → 저위험 |
| DEMO-07 | 측정 드묾 (3.5시간 간격, 결측 많음) | 결측에도 정상 동작 |
| DEMO-08 | 검사 기록 없음 | 의존도가 바이탈 쪽으로 |

필드 요약 (`DEMO-0X.json`):

| 필드 | 내용 |
|---|---|
| `vitals[]` | `t`, `HR`, `SBP`, `DBP`, `MAP`, `RR`, `SpO2`, `Temp_C` (결측 = null). Shock Index 는 엔진이 계산 |
| `labs[]` | `t_drawn`, `t_result`, `name`, `value`, `unit`, `ref_low`, `ref_high` |
| `cultures[]` | `t_collected`, `specimen`, `test_name`, `t_resulted` (null = pending), `organisms`, `resistant`, `comment` |
| `imaging[]` | `t_exam`, `t_report`, `impression` (직접 작성한 영문 문장) |
| `story_onset_hour` | 이야기상 가정한 발병 시각 (모델 입력 아님, 설명용) |
