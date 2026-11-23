"""
build_data_js.py — 엔진 결과(results/*.json) + 가상 환자 입력(patients/*.json) → 대시보드용 data.js / meta.json

  형식: DASHBOARD_RECOMMENDATION.md 13절 스키마 v1 에 아래를 반영한 v1.1
    + patients[].inputs  : 판단 과정 탭 ①(입력 시간축)용 원래 기록 (바이탈·검사·영상·배양 이벤트)
    + meta.thresholds 에 에이전트별(vital / emr) 기준 추가
    − scenarios          : what-if 화면 삭제로 생성하지 않음
    − intended_tier, story_onset_hour : 내부 설계 메모라 제외
  실행: python build_data_js.py [--out ../dashboard_handoff/data]
"""
import os, sys, json, glob, argparse, datetime

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "..", "dashboard_handoff", "data"))
a = ap.parse_args()
CFG = json.load(open(os.path.join(HERE, "config", "engine_config.json"), encoding="utf-8"))


def rnd(x):
    """float 은 4자리로 줄여 용량 축소"""
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [rnd(v) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    return x


patients, trajectories = [], {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "DEMO-*.json"))):
    res = json.load(open(f, encoding="utf-8"))
    pid = res["patient"]["patient_id"]
    src = json.load(open(os.path.join(HERE, "patients", f"{pid}.json"), encoding="utf-8"))
    patients.append({
        "id": pid, "name": src["name"], "age": src["demo"]["age"], "gender": src["demo"]["gender"],
        "scenario": src["scenario"], "description": src["description"],
        "timeline": src["timeline_hours"], "default_t": src["default_t"],
        "inputs": rnd({k: src[k] for k in ("vitals", "labs", "imaging", "cultures")}),
    })
    trajectories[pid] = {t: rnd(r) for t, r in sorted(res["results"].items(), key=lambda kv: int(kv[0]))}

th = CFG["thresholds"]
meta = {
    "schema_version": "1.1",
    "generated_at": datetime.date.today().isoformat(),
    "synthetic": True,
    "disclaimer": "가상 환자 · 미리 계산된 결과 · 연구용 (진단 대체 불가) · 점수는 보정 전 상대 위험 점수",
    "models": {"vital": "바이탈 에이전트 v3 (2층 LSTM + attention, 8피처 × 24행)",
               "emr": "EMR 에이전트 (PubMedBERT 전체 미세조정, 영문 스냅샷 문서 입력)",
               "fusion": "합동 심문 (3토큰 [CLS, vital, emr] self-attention, seed 1)",
               "physician": "책임 의사 모듈 (규칙 플래그: lab 5종 · qSOFA 2항목, SOFA 점수·감염 의심은 계산하지 않음)"},
    "thresholds": {"high": th["fusion"]["high"], "moderate": th["fusion"]["moderate"],
                   "agent": {"vital": {"high": th["vital"]["high"], "moderate": th["vital"]["moderate"]},
                             "emr": {"high": th["emr"]["high"], "moderate": th["emr"]["moderate"]}}},
    "horizon_hours": 4,
    "conventions": {
        "score_display": "fusion_score × 100 을 정수로 '위험 점수' 로 표시 (% 아님)",
        "conflict": "한 에이전트 HIGH + 다른 에이전트 LOW (각자 기준으로 등급화)",
        "agent_dependency": "융합 CLS attention 중 vital·emr 비율 (합 1). '근거 참고 비율' 로 표현, 기여도로 단정하지 않기",
        "emr_section_effect": "해당 섹션을 문서에서 뺐을 때의 EMR 점수 변화. 음수 = 그 섹션이 점수를 올리고 있었음 (판단을 이끈 근거). null = 섹션 없음 또는 미계산",
        "vital_window": "직전 24개 측정 칸 (h = 시점 T 기준 상대 시간, 앞쪽 null = 기록 없음). vital_attention 과 같은 순서",
        "inputs_time": "모든 시간은 ICU 입실 후 경과 시간(시간 단위). 시점 T 에서는 t ≤ T 인 측정과, 결과 저장 시각(t_result / t_report) ≤ T 인 검사·영상만 쓰임",
    },
    "removed_from_v1": ["scenarios (what-if 화면 없음)"],
}
th_fus = th["fusion"]
model_info = {
    "test": {"fusion_auroc": CFG["model_info"]["test_auroc"]["fusion"], "dataset": "MIMIC-IV ICU (실제 데이터) 평가"},
    "tiers": [{"tier": "HIGH", "min_score": th_fus["high"], "action": "책임의사 즉시 확인 (3시간 안 확인 권고)",
               "basis": "검증 데이터 특이도 95% 지점"},
              {"tier": "MODERATE", "min_score": th_fus["moderate"], "action": "관찰 목록 · 추가 검사 고려",
               "basis": "검증 데이터 특이도 85% 지점"},
              {"tier": "LOW", "min_score": 0, "action": "일상 모니터링"}],
    "how_it_works": ["매 시간 시점 T 에서, 그때까지 알 수 있던 정보만 사용 (결과가 저장되기 전 검사는 쓰지 않음)",
                     "바이탈은 시계열 모델이, 검사·영상은 한 장의 문서로 만들어 의료 언어모델이 읽음",
                     "두 판단을 attention 으로 교차 검증하고, Sepsis-3 관련 지표 규칙과 대조해 리포트 생성"],
    "limitations": ["가상 환자 시연 (실제 환자 아님)", "점수는 보정 전 상대 점수", "단일 기관 데이터로 학습", "연구용, 진단을 대체하지 않음"],
}
data = {"meta": meta, "patients": patients, "trajectories": trajectories, "model_info": model_info}

os.makedirs(a.out, exist_ok=True)
js = "// SepsisGuard 대시보드 데이터 (가상 환자, 미리 계산된 결과). build_data_js.py 로 생성 — 직접 수정하지 말 것\n" \
     "window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
open(os.path.join(a.out, "data.js"), "w", encoding="utf-8").write(js)
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_t = sum(len(v) for v in trajectories.values())
print(f"data.js: 환자 {len(patients)}명 · 시점 결과 {n_t}개 · {os.path.getsize(os.path.join(a.out, 'data.js')) / 1e6:.2f} MB → {os.path.abspath(a.out)}")
