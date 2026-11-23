"""results/*.json 의 guideline(규칙 판정)·report_md 만 다시 생성 (과거 점수 경과 포함) (모델 재실행 없음 → 점수·등급 그대로).
사용: python regen_reports.py
"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sepsis_system_final import report_md, SepsisEngine, CFG  # noqa: E402

# 규칙 판정(guideline)은 모델 없이 환자 파일에서 다시 계산 (모델 점수는 그대로)
eng = object.__new__(SepsisEngine)
eng.feat = CFG["vital"]["features"]; eng.W = CFG["vital"]["window_rows"]

for f in sorted(glob.glob(os.path.join(HERE, "results", "DEMO-*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    pid = d["patient"]["patient_id"]
    p = json.load(open(os.path.join(HERE, "patients", f"{pid}.json"), encoding="utf-8"))
    hist_all = sorted((int(float(t)), round(x["fusion_score"] * 100), x["tier"]) for t, x in d["results"].items())
    for T, r in d["results"].items():
        t = int(float(T))
        rows = eng.vital_rows(p, t)[-eng.W:]      # vital_tensors 와 같은 창
        labs = eng.emr_aggregates(p, t)[0]
        r["guideline"] = SepsisEngine.guideline(labs, rows, t, eng.W)
        hist = [h for h in hist_all if h[0] <= float(T)]   # 그 시점 이전 기록만 (미래 점수 사용 금지)
        r["report_md"] = report_md(r, p, hist)
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(pid, len(d["results"]), "reports")
