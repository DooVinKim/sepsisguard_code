"""
verify_engine.py — 최종 엔진이 학습 파이프라인과 같은 값을 내는지 검증 (실제 test 데이터 사용, 집계 숫자만 출력)

  [1] 문서 직렬화 : 학습 때 만든 윈도잉 중간 산출물(parquet) → 엔진의 직렬화 함수 → 학습 문서(emr_text)와 글자 단위 비교
  [2] 바이탈       : 학습 때 정규화된 시간 칸 행 → 엔진의 바이탈 모델 호출 → 융합 데이터의 vital_prob 와 비교
  [3] 전체 연결    : 학습 문서 + [2] 의 바이탈 벡터 → 엔진의 EMR·융합 → 융합 예측 파일(seed 1)과 비교
  ※ 원본 환자 텍스트·ID 는 출력하지 않는다. 불일치는 섹션별 개수만 보고한다.

실행: python verify_engine.py [--n_doc 3000] [--n_vital 500] [--n_chain 200]
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sepsis_system_final as S

ap = argparse.ArgumentParser()
ap.add_argument("--n_doc", type=int, default=3000)
ap.add_argument("--n_vital", type=int, default=500)
ap.add_argument("--n_chain", type=int, default=200)
a = ap.parse_args()
R = str(S.ROOT); V2 = f"{R}/data/v2"
rng = np.random.default_rng(0)

docs = pq.read_table(f"{V2}/emr_documents_b0_4_relax_v3_L48.parquet", columns=["stay_id", "t_point", "label", "split", "emr_text"]).to_pandas()
docs = docs[docs.split == "test"].reset_index(drop=True)
docs["tk"] = pd.to_datetime(docs.t_point, utc=True).astype("int64") // 10**9

# ════════ [1] 문서 직렬화 ════════
t0 = time.time()
smp = docs.iloc[rng.choice(len(docs), min(a.n_doc, len(docs)), replace=False)].reset_index(drop=True)
keys = set(zip(smp.stay_id, smp.tk))
stays = list(set(smp.stay_id))

def load(name, cols):
    t = pq.read_table(f"{V2}/{name}", columns=cols, filters=[("stay_id", "in", stays)]).to_pandas()
    t["tk"] = pd.to_datetime(t.t_point, utc=True).astype("int64") // 10**9
    return t[[k in keys for k in zip(t.stay_id, t.tk)]]

labs = load("emr_labs_windowed_b0_4_relax_L48.parquet",
            ["stay_id", "t_point", "lab_name", "last_val", "uom", "delta_val", "n_meas", "any_abnormal", "ref_low", "ref_high", "vals", "hrs_ago"])
micro = load("emr_micro_windowed_b0_4_relax.parquet", ["stay_id", "t_point", "n_cultures", "blood_culture", "specimen_types"])
mres = load("emr_micro_results_b0_4_relax_M72.parquet",
            ["stay_id", "t_point", "spec_type_desc", "test_name", "collect_time", "result_time", "status", "organisms", "resistant", "comment", "cancelled"])
radio = load("emr_radio_windowed_b0_4_relax.parquet", ["stay_id", "t_point", "note_time", "text"])

def nn_(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x

L = {}
for r in labs.itertuples(index=False):
    L.setdefault((r.stay_id, r.tk), []).append({"lab_name": r.lab_name, "n_meas": int(r.n_meas), "vals": list(r.vals), "hrs_ago": list(r.hrs_ago),
        "uom": nn_(r.uom), "ref_low": nn_(r.ref_low), "ref_high": nn_(r.ref_high), "last_val": r.last_val, "delta_val": r.delta_val,
        "any_abnormal": bool(r.any_abnormal)})
M = {(r.stay_id, r.tk): {"n_cultures": int(r.n_cultures), "blood_culture": bool(r.blood_culture), "specimen_types": nn_(r.specimen_types)} for r in micro.itertuples(index=False)}
RS = {}
for r in mres.itertuples(index=False):
    if not (r.cancelled is False or pd.isna(r.cancelled) or nn_(r.organisms) is not None):
        continue
    tp = pd.Timestamp(r.t_point); tp = tp.tz_localize(None) if tp.tzinfo else tp
    ct = pd.Timestamp(r.collect_time); rt = pd.Timestamp(r.result_time) if not pd.isna(r.result_time) else None
    RS.setdefault((r.stay_id, r.tk), []).append({"spec_type_desc": r.spec_type_desc, "test_name": r.test_name,
        "c_ago": (tp - ct).total_seconds() / 3600, "r_ago": None if rt is None else (tp - rt).total_seconds() / 3600,
        "status": r.status, "organisms": nn_(r.organisms), "resistant": nn_(r.resistant), "comment": nn_(r.comment)})
RD = {}
for r in radio.itertuples(index=False):
    tp = pd.Timestamp(r.t_point); tp = tp.tz_localize(None) if tp.tzinfo else tp
    RD[(r.stay_id, r.tk)] = {"hrs_ago": (tp - pd.Timestamp(r.note_time)).total_seconds() / 3600, "text": r.text}

same = 0; diff_sec = {"labs": 0, "drawn": 0, "results": 0, "radio": 0}
for r in smp.itertuples(index=False):
    k = (r.stay_id, r.tk)
    blocks = S.blocks_from_aggregates(L.get(k, []), M.get(k), RS.get(k, []), RD.get(k))
    doc = S.assemble(*blocks)
    if doc == r.emr_text:
        same += 1; continue
    a_ = doc.split("\n\n"); b_ = r.emr_text.split("\n\n")
    for name, i in zip(["labs", "drawn", "results", "radio"], range(4)):
        if i >= len(a_) or i >= len(b_) or a_[i] != b_[i]:
            diff_sec[name] += 1
print(f"[1] 문서 직렬화: {len(smp)}개 중 완전 일치 {same} ({same / len(smp) * 100:.2f}%) · 불일치 섹션 {diff_sec} · {time.time() - t0:.0f}초")

# ════════ [2] 바이탈 ════════
t0 = time.time()
eng = S.SepsisEngine(load_emr=a.n_chain > 0)
FEAT = eng.feat; W = eng.W
fz = np.load(f"{R}/data/fusion_data_pubmedbert_pw5/test.npz", allow_pickle=True)
fkey = pd.Series(np.arange(len(fz["label"])), index=pd.MultiIndex.from_arrays(
    [fz["stay_id"].astype(np.int64), pd.to_datetime(pd.Series(fz["t_point"].astype(str)), utc=True).astype("int64").to_numpy() // 10**9]))
vs = docs.iloc[rng.choice(len(docs), a.n_vital, replace=False)].reset_index(drop=True)
wide = pq.read_table(f"{V2}/processed_labeled_b0_4_relax.parquet", columns=["subject_id", "stay_id", "hour_bucket"] + FEAT,
                     filters=[("stay_id", "in", list(set(vs.stay_id)))]).to_pandas()
pat = pd.read_csv(f"{R}/data/patients.csv", usecols=["subject_id", "gender", "anchor_age"])
wide = wide.merge(pat, on="subject_id", how="left")
wide["ep"] = pd.to_datetime(wide.hour_bucket, utc=True).astype("int64") // 10**9
wide = wide.sort_values(["stay_id", "ep"])
G = {sid: g for sid, g in wide.groupby("stay_id")}
X = np.full((len(vs), W, len(FEAT)), S.MASK, np.float32); PAD = np.ones((len(vs), W), bool); DEMO = np.zeros((len(vs), 2), np.float32); ok = np.zeros(len(vs), bool)
for i, r in enumerate(vs.itertuples(index=False)):
    g = G.get(r.stay_id)
    if g is None: continue
    eps = g.ep.to_numpy(); pos = int(np.searchsorted(eps, r.tk))
    if pos >= len(eps) or eps[pos] != r.tk: continue
    seg = g[FEAT].to_numpy(np.float32)[max(0, pos - W + 1):pos + 1]
    X[i, W - len(seg):] = seg; PAD[i, W - len(seg):] = False
    age = g.anchor_age.iloc[0]
    DEMO[i] = [0.0 if pd.isna(age) else (age - S.CFG["vital"]["age_mean"]) / S.CFG["vital"]["age_sd"], {"M": 1.0, "F": 0.0}.get(g.gender.iloc[0], 0.5)]
    ok[i] = True
vp, vctx, _ = eng.run_vital(X[ok], PAD[ok], DEMO[ok])
idx = fkey.reindex(pd.MultiIndex.from_arrays([vs.stay_id[ok].astype(np.int64), vs.tk[ok]])).to_numpy()
m = ~np.isnan(idx); idx = idx[m].astype(int)
ref = fz["vital_prob"][idx]; got = vp[m]
print(f"[2] 바이탈: {m.sum()}개 · 상관 {np.corrcoef(got, ref)[0, 1]:.6f} · 최대 절대 차이 {np.abs(got - ref).max():.6f} · {time.time() - t0:.0f}초")

# ════════ [3] 전체 연결 (EMR + 융합) ════════
if a.n_chain > 0:
    t0 = time.time()
    ch = np.where(m)[0][:a.n_chain]
    texts = vs.emr_text[ok].to_numpy()[ch].tolist()
    ep, evec = eng.run_emr(texts)
    fp, _ = eng.run_fusion(vctx[ch], evec)
    pz = np.load(f"{R}/data/fusion_pred_pubmedbert_pw5_s1.npz", allow_pickle=True)
    pkey = pd.Series(np.arange(len(pz["test_prob"])), index=pd.MultiIndex.from_arrays(
        [pz["stay_id"].astype(np.int64), pd.to_datetime(pd.Series(pz["t_point"].astype(str)), utc=True).astype("int64").to_numpy() // 10**9]))
    sel = vs[ok].iloc[ch]
    pi = pkey.reindex(pd.MultiIndex.from_arrays([sel.stay_id.astype(np.int64), sel.tk])).to_numpy().astype(int)
    print(f"[3] 전체 연결: {len(ch)}개 · EMR 확률 상관 {np.corrcoef(ep, fz['emr_prob'][idx[:len(ch)]])[0, 1]:.5f} · "
          f"융합 상관 {np.corrcoef(fp, pz['test_prob'][pi])[0, 1]:.5f} · 융합 최대 차이 {np.abs(fp - pz['test_prob'][pi]).max():.4f} · {time.time() - t0:.0f}초")
