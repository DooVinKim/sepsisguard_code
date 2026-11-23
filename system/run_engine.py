"""
run_engine.py — 가상 환자를 최종 엔진으로 판정 (직접 실행용)

  python run_engine.py                         # 8명 전원, 위험 추이 24~48h (섹션 기여도는 마지막 시점만) · 약 10~15분 (CPU)
  python run_engine.py --patient DEMO-01       # 한 명만
  python run_engine.py --patient DEMO-01 --t 48 --report    # 한 시점만 + 책임의사 리포트 출력
  python run_engine.py --sections none         # 섹션 기여도 계산 생략 (더 빠름)

  출력: results/DEMO-0X.json  (시점별 Result, 대시보드·공개 데모 입력) + 화면 요약표
"""
import os, sys, json, glob, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")       # PubMedBERT 기본 구조는 이미 캐시에 있음 (인터넷 불필요)
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sepsis_system_final import SepsisEngine, TIER_KO

ap = argparse.ArgumentParser()
ap.add_argument("--patient", default="all", help="DEMO-01 같은 ID, 또는 all")
ap.add_argument("--t", type=int, default=None, help="한 시점만 (예: 48). 생략하면 timeline 전체")
ap.add_argument("--sections", default="last", choices=["all", "last", "none"], help="EMR 섹션 기여도 계산 범위")
ap.add_argument("--report", action="store_true", help="마지막 시점의 책임의사 리포트를 화면에 출력")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(HERE, "patients", "DEMO-*.json")))
if a.patient != "all":
    files = [f for f in files if os.path.basename(f).startswith(a.patient)]
assert files, f"환자 파일 없음: {a.patient}"
os.makedirs(a.out, exist_ok=True)

t0 = time.time()
eng = SepsisEngine()
print(f"엔진 로드 {time.time() - t0:.0f}초 (device={eng.dev})\n")

summary = []
for f in files:
    p = json.load(open(f, encoding="utf-8")); pid = p["patient_id"]
    print(f"── {pid} · {p['scenario']} · 의도: {p['intended_tier']}")
    print(f"   {'T':>3} | {'점수':>4} | {'등급':<8} | {'바이탈':>6} {'':<8} | {'EMR':>6} {'':<8} | 충돌 | 의존도 V:E")
    t1 = time.time()

    def show(T, r):
        print(f"   {T:>3} | {round(r['fusion_score'] * 100):>4} | {TIER_KO[r['tier']]:<8} | {r['vital_score']:.3f} {TIER_KO[r['vital_tier']]:<8} | "
              f"{r['emr_score']:.3f} {TIER_KO[r['emr_tier']]:<8} | {'예' if r['conflict'] else '  '}   | "
              f"{r['agent_dependency']['vital'] * 100:.0f}:{r['agent_dependency']['emr'] * 100:.0f}", flush=True)

    if a.t is not None:
        res = {a.t: eng.assess(p, a.t, sections=(a.sections != "none"))}; show(a.t, res[a.t])
    else:
        res = eng.trajectory(p, sections=a.sections, progress=show)
    last = res[p["default_t"]] if p.get("default_t") in res else res[max(res)]   # 대시보드 기본 시점 기준
    print(f"   ({time.time() - t1:.0f}초)")
    if last["emr_section_effect"]:
        print("   섹션 기여도 (기본 시점, 빼면 EMR 점수 변화):", {k: v for k, v in last["emr_section_effect"].items() if v is not None})
    if a.report:
        print("\n" + last["report_md"] + "\n")
    json.dump({"patient": {k: p[k] for k in ("patient_id", "name", "scenario", "description", "intended_tier", "demo", "timeline_hours", "default_t")},
               "results": {str(k): v for k, v in res.items()}},
              open(os.path.join(a.out, f"{pid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summary.append((pid, p["scenario"], p["intended_tier"], TIER_KO[last["tier"]], round(last["fusion_score"] * 100),
                    min(round(r["fusion_score"] * 100) for r in res.values()), max(round(r["fusion_score"] * 100) for r in res.values())))
    print()

print("══ 요약 (기본 시점 기준) ══")
for pid, sc, intended, tier, s, lo, hi in summary:
    print(f"{pid}  {sc:<22s} 기본 시점 {tier:<5} {s:>3}점 (추이 {lo}~{hi})  ← 의도: {intended}")
print(f"\n결과 저장: {a.out}  · 총 {time.time() - t0:.0f}초")
