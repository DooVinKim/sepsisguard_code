"""
fusion_06a_v3.py — 융합용 정렬 데이터 (바이탈 v3 + EMR LoRA, preprocess_v2 b0_4_relax 그리드)
fusion_06a_win4.py 대비 변경
  - 바이탈: code/vital_v3/vital_v3_model.py (T4_relax_w24_all, W=24, 8피처)
            forward 에 pad_mask 추가. pad 는 "실측 행 없음" 기준 (값이 -1 인지로 만들지 않음)
  - 그리드: data/v2/emr_documents_<tag>_k<K>.parquet 의 (stay_id, t_point), split 별
  - 미매칭 시점은 윈도우가 전부 패딩 → context NaN 이므로 저장에서 제외하고 개수 보고
  - EMR 어댑터·문서·max_len 을 인자로 (v2 문서로 재학습된 LoRA 가 나오면 --adapter 만 교체)
출력: data/fusion_data_v3/{train,val,test}.npz
       vital_vec(128), emr_vec(768), vital_prob, emr_prob, label, stay_id, t_point
실행: python fusion_06a_v3.py [--adapter models/emr_lora_labfix] [--k 3] [--max_len 320]
"""
import os, sys, argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn
os.environ.setdefault("HF_HOME", r"./models/hf_cache")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"vital")
import vital_v3_model as vm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="b0_4_relax")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--adapter", default=r"./models/emr_lora_labfix")
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--out", default=r"./data/fusion_data_v3")
a = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V2 = r"./data/v2"
os.makedirs(a.out, exist_ok=True)
WIN, FEAT, NV = vm.WINDOW, vm.FEATURES, len(vm.FEATURES)
DELTA_T = np.arange(WIN, 0, -1, dtype=np.float32)

# ── 바이탈 v3 ──────────────────────────────────────────────
vital = vm.load(DEV)
print(f"바이탈 v3 로드: W={WIN} V={NV} {FEAT}", flush=True)

# stay별 (epoch, 바이탈행렬, demo). demo 규칙 = vital_v3.py (train 행 기준 age Z, 결측 0 / M1 F0 그외 0.5)
wide = pd.read_parquet(os.path.join(V2, f"processed_labeled_{a.tag}.parquet"),
                       columns=["subject_id", "stay_id", "hour_bucket", "split"] + FEAT)
pat = pd.read_csv(r"./data/patients.csv", usecols=["subject_id", "gender", "anchor_age"])
wide = wide.merge(pat, on="subject_id", how="left")
tr_age = wide.loc[wide.split == "train", "anchor_age"]; mu, sd = tr_age.mean(), tr_age.std() or 1.0
wide["age_z"] = ((wide.anchor_age - mu) / sd).fillna(0.0)
wide["gender_bin"] = wide.gender.map({"M": 1.0, "F": 0.0}).fillna(0.5)
print(f"age_z: mean={mu:.4f} sd={sd:.4f}", flush=True)
wide["epoch"] = wide["hour_bucket"].astype("int64") // 10**9
wide = wide.sort_values(["stay_id", "epoch"])
stay_grp = {}
for sid, g in wide.groupby("stay_id", sort=False):
    stay_grp[sid] = (g["epoch"].to_numpy(), g[FEAT].to_numpy(np.float32),
                     g[vm.DEMO].to_numpy(np.float32)[0])
del wide
print(f"stay 인덱스: {len(stay_grp):,}", flush=True)

def build_windows(stay_ids, epochs):
    """각 (stay_id, epoch) → 직전 WIN 행. 앞 부족분은 -1 행 + pad=True. ok=False 는 미매칭."""
    W = np.full((len(stay_ids), WIN, NV), vm.MASK, np.float32)
    P = np.ones((len(stay_ids), WIN), bool)
    D = np.zeros((len(stay_ids), len(vm.DEMO)), np.float32)
    ok = np.zeros(len(stay_ids), bool)
    for k, (sid, ep) in enumerate(zip(stay_ids, epochs)):
        g = stay_grp.get(int(sid))
        if g is None: continue
        eps, mat, demo = g
        pos = int(np.searchsorted(eps, ep))
        if pos >= len(eps) or eps[pos] != ep: continue
        lo = max(0, pos - WIN + 1); seg = mat[lo:pos + 1]
        W[k, WIN - len(seg):] = seg; P[k, WIN - len(seg):] = False
        D[k] = demo; ok[k] = True
    return W, P, D, ok

@torch.no_grad()
def vital_ctx(W, P, D):
    C, Pr = [], []
    for i in range(0, len(W), 1024):
        vx = torch.from_numpy(W[i:i+1024]).to(DEV)
        pad = torch.from_numpy(P[i:i+1024]).to(DEV)
        dx = torch.from_numpy(np.repeat(D[i:i+1024][:, None, :], WIN, 1)).to(DEV)
        dt = torch.from_numpy(np.tile(DELTA_T, (len(vx), 1))).to(DEV)
        o = vital(vx, pad, dx, dt)
        C.append(o["context"].cpu().numpy()); Pr.append(o["sepsis_prob"].squeeze(-1).cpu().numpy())
    return np.concatenate(C), np.concatenate(Pr)

# ── EMR LoRA ───────────────────────────────────────────────
MDIR = r"./models/bio_clinicalbert"
tok = AutoTokenizer.from_pretrained(MDIR)
class EmrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = PeftModel.from_pretrained(AutoModel.from_pretrained(MDIR), a.adapter)
        self.head = nn.Linear(768, 1)
        self.head.load_state_dict(torch.load(os.path.join(a.adapter, "head.pt"), map_location=DEV))
    def forward(self, enc):
        h = self.bert(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1), pooled
emr = EmrModel().to(DEV).eval()
print(f"EMR LoRA 로드: {a.adapter}", flush=True)

@torch.no_grad()
def emr_ctx(texts, batch=64):
    # 길이순 정렬로 패딩 낭비 줄이고 원래 순서로 복원
    order = np.argsort([len(t) for t in texts]); P = np.zeros(len(texts), np.float32); V = np.zeros((len(texts), 768), np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i+batch]
        enc = tok([texts[j] for j in idx], return_tensors="pt", truncation=True, max_length=a.max_len, padding=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        with torch.autocast(DEV, dtype=torch.float16):
            logit, pooled = emr(enc)
        P[idx] = torch.sigmoid(logit).float().cpu().numpy(); V[idx] = pooled.float().cpu().numpy()
    return P, V

# ── split별 정렬 ────────────────────────────────────────────
from sklearn.metrics import roc_auc_score
docs = pd.read_parquet(os.path.join(V2, f"emr_documents_{a.tag}_k{a.k}.parquet"))
report = {"tag": a.tag, "k": a.k, "adapter": a.adapter, "max_len": a.max_len,
          "vital_ckpt": vm.CKPT, "window": WIN, "features": FEAT}
for split in ["train", "val", "test"]:
    s = docs[docs["split"] == split].reset_index(drop=True)
    stay_id = s["stay_id"].to_numpy(); label = s["label"].to_numpy()
    t_str = s["t_point"].astype(str).to_numpy()
    epochs = (pd.to_datetime(s["t_point"], utc=True).astype("int64") // 10**9).to_numpy()

    W, P, D, ok = build_windows(stay_id, epochs)
    n_miss = int((~ok).sum())
    if n_miss: print(f"  [{split}] 미매칭 {n_miss} 시점 제외 (전부 패딩 → NaN)", flush=True)
    W, P, D, s = W[ok], P[ok], D[ok], s[ok].reset_index(drop=True)
    stay_id, label, t_str = stay_id[ok], label[ok], t_str[ok]

    vvec, vprob = vital_ctx(W, P, D)
    eprob, evec = emr_ctx(s["emr_text"].tolist())
    assert not np.isnan(vvec).any() and not np.isnan(evec).any()

    np.savez_compressed(os.path.join(a.out, f"{split}.npz"),
        vital_vec=vvec.astype(np.float32), emr_vec=evec.astype(np.float32),
        vital_prob=vprob.astype(np.float32), emr_prob=eprob.astype(np.float32),
        label=label.astype(np.int64), stay_id=stay_id.astype(np.int64), t_point=t_str)
    r = {"N": int(len(label)), "pos": int(label.sum()), "miss": n_miss,
         "pad_rate": float(P.any(1).mean()),
         "vital_auroc": float(roc_auc_score(label, vprob)), "emr_auroc": float(roc_auc_score(label, eprob))}
    report[split] = r
    print(f"[{split}] N={r['N']:,} pos={r['pos']:,} 미매칭{n_miss} 패딩윈도우={r['pad_rate']*100:.1f}% "
          f"| 단독 AUROC vital={r['vital_auroc']:.4f} emr={r['emr_auroc']:.4f}", flush=True)

json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=2, ensure_ascii=False)
print(f"\n=== 6a(v3) 완료: {a.out} ===")
