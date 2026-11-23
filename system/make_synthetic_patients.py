"""
make_synthetic_patients.py — 시연용 가상(합성) 환자 생성 (실제 환자 데이터 사용 안 함)

  - 시나리오별 "흐름"(시간 → 값 기준점)을 정의하면 시간별 값을 선형 보간 + 작은 잡음으로 생성
  - 형식: DASHBOARD_RECOMMENDATION.md 4절 "입력 형식 v2" (시간 = 입원 후 경과 시간, 실제 날짜 없음)
  - 검사 이름·단위·정상범위, 배양 검체·검사명 표기는 학습 문서와 같은 표기를 따름 (집계로 확인한 대표값)
  - 같은 seed 면 항상 같은 결과. 등급 조정이 필요하면 아래 SCENARIOS 의 기준점만 고쳐 다시 실행

실행: python make_synthetic_patients.py            → patients/DEMO-0X.json, patients/index.json
"""
import json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "patients")
SEED = 2026
STAY_END = 49                  # 바이탈 기록 0~48시간 (48시 칸 포함: 학습처럼 시점 T 의 시간 칸에 측정이 있도록)
TIMELINE = [24, 48]            # 위험 추이를 계산할 시점 범위 (1시간 간격)

# 학습 문서의 대표 단위·정상범위 (labevents 집계 최빈값)
LAB_META = {
    "Lactate":     ("mmol/L", 0.5, 2.0),
    "WBC":         ("K/uL", 4.0, 11.0),
    "Neutrophils": ("%", 50.0, 70.0),
    "Bands":       ("%", 0.0, 5.0),
    "Platelet":    ("K/uL", 150.0, 400.0),
    "Creatinine":  ("mg/dL", 0.5, 1.2),
    "BUN":         ("mg/dL", 6.0, 20.0),
    "Bilirubin":   ("mg/dL", 0.0, 1.5),
    "Bicarbonate": ("mEq/L", 22.0, 32.0),
    "CRP":         ("mg/L", 0.0, 5.0),
}
VITAL_NOISE = {"HR": 3.0, "SBP": 4.0, "DBP": 3.0, "RR": 1.2, "SpO2": 0.6, "Temp_C": 0.12}


def curve(points, hours):
    """기준점 [(시간, 값), ...] 을 선형 보간"""
    xs, ys = zip(*points)
    return np.interp(hours, xs, ys)


def make_vitals(rng, spec):
    """시간별 바이탈. spec: 바이탈별 기준점 + 측정 간격·결측 설정"""
    hours = np.arange(spec.get("start", 0), STAY_END, spec.get("every", 1.0))
    rows = []
    for h in hours:
        r = {"t": round(float(h) + float(rng.uniform(0.0, 0.25)), 2)}   # 측정 시각이 정각에서 조금씩 어긋남
        for k in ["HR", "SBP", "DBP", "RR", "SpO2", "Temp_C"]:
            v = curve(spec[k], h) + rng.normal(0, VITAL_NOISE[k])
            r[k] = round(float(v), 1) if k == "Temp_C" else int(round(v))
        r["SpO2"] = min(r["SpO2"], 100)
        if h % spec.get("temp_every", 4) != 0:                 # 체온은 몇 시간마다만 기록 (실제 기록 관행)
            r["Temp_C"] = None
        if rng.random() < spec.get("bp_missing", 0.08):          # 혈압 결측 (커프 미측정)
            r["SBP"] = r["DBP"] = None
        if r["SBP"] is not None:
            r["MAP"] = int(round((r["SBP"] + 2 * r["DBP"]) / 3))
        else:
            r["MAP"] = None
        rows.append(r)
    return rows


def make_labs(spec):
    """검사: {이름: [(채혈 시각, 값[, 결과까지 걸린 시간]), ...]}. 기본 결과 저장은 채혈 후 0.5~1.5시간"""
    out = []
    for name, pts in spec.items():
        unit, lo, hi = LAB_META[name]
        for i, pt in enumerate(pts):
            t, v = pt[0], pt[1]
            delay = pt[2] if len(pt) > 2 else (0.5 if name == "Lactate" else (1.0 + 0.5 * (i % 2)))
            out.append({"t_drawn": float(t), "t_result": float(t) + delay, "name": name,
                        "value": v, "unit": unit, "ref_low": lo, "ref_high": hi})
    return sorted(out, key=lambda x: (x["t_drawn"], x["name"]))


def culture(t_collected, specimen, test_name, t_resulted=None, organisms=None, resistant=None, comment=None):
    return {"t_collected": float(t_collected), "specimen": specimen, "test_name": test_name,
            "t_resulted": None if t_resulted is None else float(t_resulted),
            "organisms": organisms, "resistant": resistant, "comment": comment}


def imaging(t_exam, t_report, impression):
    return {"t_exam": float(t_exam), "t_report": float(t_report), "impression": impression}


BLOOD = ("BLOOD CULTURE", "Blood Culture, Routine")
URINE = ("URINE", "URINE CULTURE")
SPUTUM = ("SPUTUM", "RESPIRATORY CULTURE")
MRSA = ("MRSA SCREEN", "MRSA SCREEN")

# ══════════════════════════════════════════════════════════════════════
# 시나리오 (모든 값은 지어낸 가상 값)
#   intended_tier: 의도한 마지막 시점 등급 → 모델로 돌려 확인·조정 필요
# ══════════════════════════════════════════════════════════════════════
SCENARIOS = [
    {
        "id": "DEMO-01", "age": 71, "gender": "M", "intended_tier": "HIGH (점진 상승)",
        "scenario": "폐렴 진행",
        "description": "발열·빈호흡이 시작되고 젖산·백혈구가 오르며, 폐렴 판독 후 혈압이 떨어진다. 점수가 시간에 따라 저위험에서 고위험으로 오르는 장면",
        "story_onset_hour": 52,
        "vitals": {"HR": [(0, 84), (24, 92), (36, 108), (48, 122)], "SBP": [(0, 128), (30, 118), (40, 104), (48, 91)],
                   "DBP": [(0, 72), (30, 66), (48, 52)], "RR": [(0, 17), (24, 20), (36, 25), (48, 29)],
                   "SpO2": [(0, 97), (30, 95), (48, 91)], "Temp_C": [(0, 37.0), (24, 37.6), (36, 38.4), (48, 38.9)]},
        "labs": {"Lactate": [(6, 1.3), (30, 2.4), (44, 3.8)], "WBC": [(6, 9.4), (30, 14.6), (44, 17.8)],
                 "Neutrophils": [(6, 70.0), (30, 82.0), (44, 88.0)], "Bands": [(44, 9.0)],
                 "Platelet": [(6, 228.0), (30, 176.0), (44, 141.0)], "Creatinine": [(6, 0.9), (30, 1.3), (44, 1.7)],
                 "BUN": [(6, 17.0), (30, 25.0), (44, 32.0)], "Bilirubin": [(6, 0.8), (44, 1.2)],
                 "Bicarbonate": [(6, 25.0), (30, 22.0), (44, 18.0)], "CRP": [(30, 118.0)]},
        "cultures": [culture(0.5, *BLOOD)],                                            # 입실 시 혈액배양 · 48h 까지 결과 대기
        "imaging": [imaging(3, 5, "No acute cardiopulmonary process."),
                    imaging(34, 36.5, "New right lower lobe consolidation concerning for pneumonia. Small right pleural effusion.")],
    },
    {
        "id": "DEMO-02", "age": 79, "gender": "F", "default_t": 41, "intended_tier": "MODERATE → HIGH (젖산 결과가 나오는 시점부터 반영)",
        "scenario": "요로감염 의심 (검사 결과 도착)",
        "description": "38시간에 채혈한 젖산 결과가 41시간에 나온다. 결과가 저장되기 전 시점에는 판단에 쓰이지 않고, 저장된 시점부터 반영되는 '시점 기준 판단' 을 보여주는 장면",
        "story_onset_hour": 50,
        "vitals": {"HR": [(0, 92), (24, 98), (40, 108), (48, 114)], "SBP": [(0, 116), (30, 108), (48, 97)],
                   "DBP": [(0, 64), (48, 54)], "RR": [(0, 19), (36, 22), (48, 25)],
                   "SpO2": [(0, 96), (48, 94)], "Temp_C": [(0, 37.9), (12, 38.4), (24, 37.8), (40, 38.6), (48, 38.8)]},
        "labs": {"Lactate": [(4, 1.8), (26, 2.1), (38, 4.2, 3.0), (45, 4.8)], "WBC": [(4, 12.8), (26, 14.1), (38, 17.2, 3.0), (45, 18.4)],
                 "Neutrophils": [(4, 79.0), (38, 87.0, 3.0)], "Platelet": [(4, 212.0), (26, 198.0), (38, 158.0, 3.0)],
                 "Creatinine": [(4, 1.4), (26, 1.6), (38, 2.1, 3.0), (45, 2.4)], "BUN": [(4, 29.0), (26, 34.0), (38, 41.0, 3.0), (45, 44.0)],
                 "Bicarbonate": [(4, 23.0), (38, 19.0, 3.0)]},
        "cultures": [culture(0.5, *BLOOD), culture(0.5, *URINE, t_resulted=30, organisms="ESCHERICHIA COLI")],
        "imaging": [imaging(22, 24.5, "Mild left hydronephrosis with perinephric fat stranding, suggestive of pyelonephritis. No obstructing stone.")],
    },
    {
        "id": "DEMO-03", "age": 58, "gender": "M", "default_t": 40, "intended_tier": "충돌 (바이탈 높음, EMR 낮음)",
        "scenario": "바이탈만 나쁨 (수술 후 통증·탈수 설정)",
        "description": "빈맥·약간 낮은 혈압이 계속되지만 검사는 정상 범위. 바이탈과 EMR 의 판단이 갈리는 '합동 심문' 장면",
        "story_onset_hour": None,
        "vitals": {"HR": [(0, 118), (24, 126), (40, 130), (48, 128)], "SBP": [(0, 100), (24, 94), (48, 92)],
                   "DBP": [(0, 58), (48, 52)], "RR": [(0, 22), (48, 25)],
                   "SpO2": [(0, 96), (48, 96)], "Temp_C": [(0, 37.3), (48, 37.5)]},
        "labs": {"Lactate": [(4, 1.3), (28, 1.2)], "WBC": [(4, 8.8), (28, 9.1)], "Neutrophils": [(4, 66.0)],
                 "Platelet": [(4, 262.0), (28, 248.0)], "Creatinine": [(4, 0.9), (28, 0.8)],
                 "BUN": [(4, 16.0), (28, 14.0)], "Bicarbonate": [(4, 25.0), (28, 26.0)], "CRP": [(4, 12.0)]},
        "cultures": [culture(0.5, *BLOOD)],                                            # 수술 후 입실 시 혈액배양 · 결과 대기
        "imaging": [imaging(2, 4, "Postoperative changes of the abdomen. No acute process.")],
    },
    {
        "id": "DEMO-04", "age": 66, "gender": "F", "intended_tier": "EMR 이 끌어올림 (바이탈 낮음, EMR 높음)",
        "scenario": "검사만 나쁨",
        "description": "바이탈은 아직 괜찮은데 젖산·백혈구·크레아티닌·빌리루빈이 오른다. EMR 이 바이탈의 빈틈을 보완하는 장면 (발표 상호보완 슬라이드와 연결)",
        "story_onset_hour": 54,
        "vitals": {"HR": [(0, 82), (48, 91)], "SBP": [(0, 126), (48, 119)], "DBP": [(0, 72), (48, 66)],
                   "RR": [(0, 17), (48, 19)], "SpO2": [(0, 97), (48, 96)], "Temp_C": [(0, 37.2), (48, 37.7)]},
        "labs": {"Lactate": [(6, 1.6), (16, 2.0), (22, 2.4), (28, 2.8), (34, 3.1), (40, 3.5), (46, 3.9)],
                 "WBC": [(6, 11.2), (16, 13.0), (22, 14.6), (28, 16.4), (34, 17.5), (40, 18.8), (46, 19.6)],
                 "Neutrophils": [(6, 76.0), (22, 83.0), (34, 88.0), (46, 91.0)], "Bands": [(34, 9.0), (46, 13.0)],
                 "Platelet": [(6, 188.0), (22, 158.0), (34, 131.0), (46, 108.0)],
                 "Creatinine": [(6, 1.1), (16, 1.3), (28, 1.5), (40, 1.8), (46, 2.0)],
                 "BUN": [(6, 22.0), (22, 27.0), (34, 33.0), (46, 39.0)], "Bilirubin": [(6, 1.1), (22, 1.5), (34, 2.0), (46, 2.6)],
                 "Bicarbonate": [(6, 23.0), (22, 21.0), (34, 19.0), (46, 17.0)], "CRP": [(16, 96.0), (34, 168.0)]},
        "cultures": [culture(0.5, *BLOOD), culture(0.5, *URINE, t_resulted=26, comment="NO GROWTH.")],
        "imaging": [imaging(31, 33, "Mild bibasilar atelectasis. No focal consolidation.")],
    },
    {
        "id": "DEMO-05", "age": 52, "gender": "M", "intended_tier": "LOW (유지)",
        "scenario": "안정 환자",
        "description": "바이탈·검사 모두 정상 범위. 오경보가 울리지 않는 장면",
        "story_onset_hour": None,
        "vitals": {"HR": [(0, 76), (48, 74)], "SBP": [(0, 128), (48, 126)], "DBP": [(0, 74), (48, 72)],
                   "RR": [(0, 15), (48, 15)], "SpO2": [(0, 98), (48, 98)], "Temp_C": [(0, 36.8), (48, 36.9)]},
        "labs": {"Lactate": [(6, 1.0)], "WBC": [(6, 7.2), (30, 6.9)], "Platelet": [(6, 244.0), (30, 251.0)],
                 "Creatinine": [(6, 0.8), (30, 0.8)], "BUN": [(6, 14.0), (30, 13.0)], "Bicarbonate": [(6, 26.0), (30, 27.0)]},
        "cultures": [culture(0.5, *MRSA, t_resulted=20, comment="No MRSA isolated.")],
        "imaging": [imaging(2, 4, "No acute cardiopulmonary process.")],
    },
    {
        "id": "DEMO-06", "age": 63, "gender": "M", "intended_tier": "HIGH → LOW (호전)", "timeline": [6, 48],
        "scenario": "치료 후 호전",
        "description": "병동에서 폐렴으로 배양을 채취한 뒤 악화되어 ICU 로 전동. 초기에는 고열·저혈압·젖산 상승이었다가 치료 후 바이탈과 검사가 회복된다. 점수와 등급이 내려가는 장면",
        "story_onset_hour": None,
        "vitals": {"HR": [(0, 124), (16, 112), (30, 96), (48, 86)], "SBP": [(0, 88), (16, 98), (30, 112), (48, 120)],
                   "DBP": [(0, 50), (30, 64), (48, 70)], "RR": [(0, 27), (20, 22), (48, 17)],
                   "SpO2": [(0, 92), (24, 95), (48, 97)], "Temp_C": [(0, 38.9), (20, 38.0), (36, 37.2), (48, 37.0)]},
        "labs": {"Lactate": [(1, 3.6), (14, 2.3), (30, 1.3)], "WBC": [(1, 18.2), (14, 15.1), (30, 10.4)],
                 "Neutrophils": [(1, 88.0), (30, 74.0)], "Bands": [(1, 10.0)],
                 "Platelet": [(1, 146.0), (30, 168.0)], "Creatinine": [(1, 1.6), (14, 1.4), (30, 1.1)],
                 "BUN": [(1, 30.0), (30, 22.0)], "Bicarbonate": [(1, 18.0), (30, 22.0)],
                 "CRP": [(1, 210.0)]},
        "cultures": [culture(-30, *BLOOD, t_resulted=-6, comment="NO GROWTH.")],  # 병동에서 폐렴 진단 시 채취 (ICU 전동 30h 전), 균 자람 없음
        "imaging": [imaging(1, 3, "Left lower lobe opacity concerning for pneumonia."),
                    imaging(34, 36, "Interval resolution of left lower lobe opacity. No new consolidation. Lungs are clear.")],
    },
    {
        "id": "DEMO-07", "age": 84, "gender": "F", "intended_tier": "LOW~MODERATE (결측 많아도 동작)",
        "scenario": "측정이 드문 환자",
        "description": "바이탈이 3~4시간마다만 기록되고 체온·혈압 결측이 많다. 결측이 많아도 정상 동작하는 장면",
        "story_onset_hour": None,
        "vitals": {"HR": [(0, 88), (48, 104)], "SBP": [(0, 118), (48, 102)], "DBP": [(0, 62), (48, 56)],
                   "RR": [(0, 18), (48, 23)], "SpO2": [(0, 95), (48, 93)], "Temp_C": [(0, 37.4), (48, 38.2)],
                   "every": 3.5, "temp_every": 999, "bp_missing": 0.35},
        "labs": {"WBC": [(10, 12.4)], "Creatinine": [(10, 1.3)], "BUN": [(10, 31.0)]},
        "cultures": [culture(10, *URINE, t_resulted=34, comment="NO GROWTH.")],
        "imaging": [],
    },
    {
        "id": "DEMO-08", "age": 45, "gender": "M", "intended_tier": "바이탈 의존 (검사 없음)",
        "scenario": "검사 기록이 거의 없는 환자",
        "description": "혈액검사·배양·영상이 없고 바이탈만 있다 (EMR 문서는 'No ...' 문장만). 에이전트 의존도가 바이탈 쪽으로 기우는 장면",
        "story_onset_hour": None,
        "vitals": {"HR": [(0, 90), (30, 100), (48, 112)], "SBP": [(0, 122), (48, 108)], "DBP": [(0, 70), (48, 62)],
                   "RR": [(0, 18), (48, 24)], "SpO2": [(0, 97), (48, 95)], "Temp_C": [(0, 37.4), (36, 38.0), (48, 38.5)]},
        "labs": {},
        "cultures": [],
        "imaging": [],
    },
]


def build(s, rng):
    v = dict(s["vitals"])
    spec = {k: v[k] for k in ["HR", "SBP", "DBP", "RR", "SpO2", "Temp_C"]}
    spec.update({k: v[k] for k in ["every", "temp_every", "bp_missing", "start"] if k in v})
    return {
        "patient_id": s["id"],
        "name": "가상환자 " + s["id"].split("-")[1],
        "note": "가상 환자 (실제 환자 아님). 모든 값은 시연용으로 지어낸 값",
        "scenario": s["scenario"],
        "description": s["description"],
        "intended_tier": s["intended_tier"],
        "story_onset_hour": s["story_onset_hour"],      # 이야기상 가정한 발병 시각 (모델 입력 아님)
        "demo": {"age": s["age"], "gender": s["gender"]},
        "timeline_hours": s.get("timeline", TIMELINE),
        "default_t": s.get("default_t", s.get("timeline", TIMELINE)[1]),   # 대시보드가 처음 보여줄 시점
        "vitals": make_vitals(rng, spec),
        "labs": make_labs(s["labs"]),
        "cultures": s["cultures"],
        "imaging": s["imaging"],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    index = []
    for s in SCENARIOS:
        p = build(s, rng)
        with open(os.path.join(OUT, f"{p['patient_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=1)
        index.append({k: p[k] for k in ["patient_id", "name", "scenario", "intended_tier", "description"]}
                     | {"age": p["demo"]["age"], "gender": p["demo"]["gender"],
                        "n_vitals": len(p["vitals"]), "n_labs": len(p["labs"]),
                        "n_cultures": len(p["cultures"]), "n_imaging": len(p["imaging"])})
        print(f"{p['patient_id']}  {p['scenario']:<24s} 바이탈 {len(p['vitals']):3d}행  검사 {len(p['labs']):2d}  "
              f"배양 {len(p['cultures'])}  영상 {len(p['imaging'])}  → {p['intended_tier']}")
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "patient_input_v2", "seed": SEED, "stay_end_hour": STAY_END,
                   "timeline_hours": TIMELINE, "patients": index}, f, ensure_ascii=False, indent=1)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
