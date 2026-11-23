"""
check_synthetic.py — 가상 환자 JSON 점검 (형식 · 생리학적 범위 · 시간 순서 · 실제 정보 혼입 여부)
실행: python check_synthetic.py
"""
import json, os, re, sys, glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
# 학습 전처리의 이상치 정제 범위 (CLAUDE.md, 체온은 섭씨 환산)
RANGE = {"HR": (20, 250), "SBP": (40, 250), "DBP": (10, 200), "RR": (4, 60), "SpO2": (50, 100), "Temp_C": (29, 43)}
LAB_RANGE = {"Lactate": (0.2, 20), "WBC": (0.1, 100), "Neutrophils": (0, 100), "Bands": (0, 60), "Platelet": (5, 1500),
             "Creatinine": (0.1, 15), "BUN": (1, 200), "Bilirubin": (0.1, 40), "Bicarbonate": (3, 50), "CRP": (0, 500)}
REQUIRED = ["patient_id", "note", "demo", "timeline_hours", "default_t", "vitals", "labs", "cultures", "imaging"]

errors = []
def err(pid, msg): errors.append(f"{pid}: {msg}")

files = sorted(glob.glob(os.path.join(HERE, "patients", "DEMO-*.json")))
for f in files:
    p = json.load(open(f, encoding="utf-8")); pid = p.get("patient_id", os.path.basename(f))
    for k in REQUIRED:
        if k not in p: err(pid, f"필드 없음 {k}")
    if not re.fullmatch(r"DEMO-\d{2}", pid): err(pid, "ID 형식")
    raw = open(f, encoding="utf-8").read()
    if re.search(r"\b(19|20|21)\d{2}-\d{2}-\d{2}\b", raw): err(pid, "실제 날짜 형식 문자열")
    if re.search(r"stay_id|subject_id|hadm_id", raw): err(pid, "실제 ID 필드명")
    t0, t1 = p["timeline_hours"]
    # 바이탈
    ts = [v["t"] for v in p["vitals"]]
    if ts != sorted(ts): err(pid, "바이탈 시간 순서")
    for v in p["vitals"]:
        for k, (lo, hi) in RANGE.items():
            x = v.get(k)
            if x is not None and not (lo <= x <= hi): err(pid, f"바이탈 범위 밖 {k}={x} @t{v['t']}")
        if v.get("SBP") is not None and v.get("DBP") is not None and v["DBP"] >= v["SBP"]: err(pid, f"DBP>=SBP @t{v['t']}")
    # 검사
    for l in p["labs"]:
        lo, hi = LAB_RANGE[l["name"]]
        if not (lo <= l["value"] <= hi): err(pid, f"검사 범위 밖 {l['name']}={l['value']}")
        if l["t_result"] < l["t_drawn"]: err(pid, f"검사 결과가 채혈보다 앞섬 {l['name']}")
    # 배양
    for c in p["cultures"]:
        if c["t_resulted"] is not None and c["t_resulted"] < c["t_collected"]: err(pid, "배양 결과가 채취보다 앞섬")
        if c["t_resulted"] is None and (c["organisms"] or c["resistant"]): err(pid, "pending 인데 결과 있음")
    for im in p["imaging"]:
        if im["t_report"] < im["t_exam"]: err(pid, "판독이 촬영보다 앞섬")

    # 요약 (가상 데이터라 출력 가능): 마지막 시점 T 기준 직전 24시간
    T = p["default_t"]
    win = [v for v in p["vitals"] if T - 24 <= v["t"] < T]
    def last(k):
        xs = [v[k] for v in win if v.get(k) is not None]; return xs[-1] if xs else None
    miss = {k: round(1 - np.mean([v.get(k) is not None for v in win]), 2) if win else None for k in ["SBP", "Temp_C"]}
    labs_avail = sorted({l["name"] for l in p["labs"] if l["t_result"] <= T and l["t_drawn"] >= T - 48})
    cult = [(c["specimen"], "결과" if c["t_resulted"] is not None and c["t_resulted"] <= T else "pending")
            for c in p["cultures"] if T - 72 <= c["t_collected"] <= T]
    print(f"{pid} T={T}h | 바이탈 24h {len(win)}행, 마지막 HR {last('HR')} SBP {last('SBP')} RR {last('RR')} "
          f"SpO2 {last('SpO2')} Temp {last('Temp_C')} | 결측 SBP {miss['SBP']} Temp {miss['Temp_C']} | "
          f"검사 {len(labs_avail)}종 | 배양 {cult} | 영상 {sum(1 for i in p['imaging'] if i['t_report'] <= T)}건")

# 시점 기준 예시 점검: DEMO-02 의 38h 채혈 젖산(결과 41h)은 T=40 에서 보이면 안 됨
p2 = json.load(open(os.path.join(HERE, "patients", "DEMO-02.json"), encoding="utf-8"))
for T in (40, 41):
    vis = [l for l in p2["labs"] if l["name"] == "Lactate" and l["t_drawn"] == 38 and l["t_result"] <= T]
    print(f"DEMO-02 T={T}h 38h 채혈 젖산 결과 보임: {bool(vis)}")

print("\n점검 결과:", "통과" if not errors else f"오류 {len(errors)}건")
for e in errors: print("  -", e)
