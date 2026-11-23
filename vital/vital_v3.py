"""
vital_v3.py — preprocess_v2 산출물용 바이탈 에이전트 학습 (자체 완결, vital_1 import 없음)

v1/v2 와의 차이
  1. 피처 목록·윈도우 길이를 인자/json 에서 읽음 (피처 순서 = vital_norm_stats_<tag>.json 키 순서)
  2. 양성 행 앞 이력이 window-1 보다 짧아도 -1 행으로 앞을 패딩해 윈도우 생성 (v1:93-96 은 건너뜀)
  3. AttentionLayer 가 패딩 시점을 softmax 에서 제외 (v1:217-219 에는 없음)
  4. 결과를 results/<run_name>.json 에 저장 (test AUROC/AUPRC, h-버킷별 AUROC, 이력길이별 AUROC, 학습된 α·β)
모델 구조·학습 설정은 v2 와 동일 (LSTM 2층 h128, Focal γ=1.5, pos_weight=neg_ratio, AdamW 1e-3/wd1e-4,
Cosine T_max=epochs, patience 12, val AUROC 기준 best). state_dict 키는 v1 과 호환.

실행 예
  python vital_v3.py --tag b2_4_strict --window 6 --run_name regress_w6          # 회귀 테스트
  python vital_v3.py --tag b0_4_relax  --window 6 --run_name relax_w6
  python vital_v3.py --tag b0_4_relax  --window 12 --run_name relax_w12
  python vital_v3.py --tag b0_4_relax  --window 24 --run_name relax_w24
  python vital_v3.py --tag b0_4_relax  --window 6 --features HR SBP RR SpO2 Temp SI --run_name relax_w6_base6
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# ── 인자 ────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--run_name", required=True)
ap.add_argument("--data_dir", default=r"./data/v2")
ap.add_argument("--patients_csv", default=r"./data/patients.csv")
ap.add_argument("--out_dir", default=r"./models/v2")
ap.add_argument("--window", type=int, default=6)
ap.add_argument("--features", nargs="*", default=None, help="생략하면 norm json 의 전체 피처")
ap.add_argument("--neg_per_stay", type=int, default=4)       # v2:20
ap.add_argument("--neg_ratio", type=float, default=6.0)      # v2:21
ap.add_argument("--epochs", type=int, default=60)            # v2:23
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--gamma", type=float, default=1.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--frac", type=float, default=1.0, help="train 환자 부분집합 비율 (해시 기반, test 는 전체)")
ap.add_argument("--neg_min_hist", type=int, default=None,
                help="음성 윈도우 끝 행의 최소 이력 행 수. 생략하면 train 양성의 최소 이력 (양성과 같은 조건)")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MASK = -1.0; W = a.window
os.makedirs(a.out_dir, exist_ok=True); os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True)
CKPT = os.path.join(a.out_dir, f"vital_{a.run_name}.pt")
RES  = os.path.join(a.out_dir, "results", f"vital_{a.run_name}.json")
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)

# ── 데이터 ──────────────────────────────────────────────────────
norm = json.load(open(os.path.join(a.data_dir, f"vital_norm_stats_{a.tag}.json")))
FEATURES = a.features or list(norm.keys())
missing = [f for f in FEATURES if f not in norm]
assert not missing, f"norm json 에 없는 피처: {missing}"
V = len(FEATURES)
df = pd.read_parquet(os.path.join(a.data_dir, f"processed_labeled_{a.tag}.parquet"))
df = df.sort_values(["stay_id", "hour_bucket"]).reset_index(drop=True)
log(f"tag={a.tag} W={W} V={V} {FEATURES} | rows={len(df):,} label1={int(df.label.sum()):,}")

# demo (v1:389-415 와 동일 규칙)
pat = pd.read_csv(a.patients_csv, usecols=["subject_id", "gender", "anchor_age"])
df = df.merge(pat, on="subject_id", how="left")
tr_age = df.loc[df.split == "train", "anchor_age"]
mu, sd = tr_age.mean(), tr_age.std() or 1.0
df["age_z"] = ((df.anchor_age - mu) / sd).fillna(0.0)
df["gender_bin"] = df.gender.map({"M": 1.0, "F": 0.0}).fillna(0.5)
DEMO = ["age_z", "gender_bin"]

# train 부분집합 (환자 해시, 00_config.R id_hash 와 동일식)
if a.frac < 1.0:
    h = ((df.subject_id.astype(np.float64) * 2654435761) % 2**32) / 2**32
    df = df[(df.split != "train") | (h < a.frac)]
    log(f"train frac={a.frac}: train 환자 {df[df.split=='train'].subject_id.nunique():,}")

def cut(vit, end):
    """end 행에서 끝나는 길이 W 윈도우. 앞이 모자라면 -1 패딩. 반환 (x (W,V), pad (W,) bool, hist_rows)"""
    s = end - W + 1
    if s >= 0:
        return vit[s:end + 1], np.zeros(W, bool), W - 1
    pad = -s
    x = np.vstack([np.full((pad, vit.shape[1]), MASK, np.float32), vit[:end + 1]])
    m = np.zeros(W, bool); m[:pad] = True
    return x, m, end

# 음성 이력 조건: 양성 윈도우 끝 행의 최소 이력(= 코호트 MIN_HIST_ROWS)과 맞춤.
# 안 맞추면 stay 초반 음성만 패딩되어 "패딩 = 음성" 지름길이 생김 (T1 첫 실행: 패딩 1,256개 전부 음성)
if a.neg_min_hist is None:
    trp = df[df.split == "train"].copy()
    trp["row"] = trp.groupby("stay_id").cumcount()
    a.neg_min_hist = int(trp.loc[trp.label == 1, "row"].min())
log(f"음성 윈도우 최소 이력 = {a.neg_min_hist}행")

def make(split, rng, train):
    X, P, D, Y, H, HR = [], [], [], [], [], []    # HR: 양성이면 hours_to_sepsis_bucket, 음성은 nan
    sub = df[df.split == split]
    for sid, g in sub.groupby("stay_id", sort=False):
        vit = g[FEATURES].to_numpy(np.float32); lab = g.label.to_numpy()
        d = g[DEMO].iloc[0].to_numpy(np.float32); hts = g.hours_to_sepsis_bucket.to_numpy()
        if lab.max() == 1:
            for end in np.where(lab == 1)[0]:
                x, m, hr = cut(vit, end); X.append(x); P.append(m); D.append(d); Y.append(1); H.append(hts[end]); HR.append(hr)
        else:
            cand = np.arange(a.neg_min_hist, len(g))
            if len(cand) == 0: continue
            k = min(a.neg_per_stay if train else 1, len(cand))
            for end in rng.choice(cand, k, replace=False):
                x, m, hr = cut(vit, end); X.append(x); P.append(m); D.append(d); Y.append(0); H.append(np.nan); HR.append(hr)
    X, P, D, Y, H, HR = map(np.array, (X, P, D, Y, H, HR))
    if train:
        pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        neg = rng.choice(neg, min(len(neg), int(len(pos) * a.neg_ratio)), replace=False)
        idx = np.concatenate([pos, neg]); X, P, D, Y, H, HR = (z[idx] for z in (X, P, D, Y, H, HR))
    pd_ = P.any(1)
    log(f"{split:5s} N={len(Y):,} pos={int(Y.sum()):,} ({Y.mean()*100:.2f}%) 패딩윈도우={pd_.mean()*100:.1f}% "
        f"(양성 {pd_[Y == 1].mean()*100:.1f}% / 음성 {pd_[Y == 0].mean()*100:.1f}%)")
    return X, P, D, Y, H, HR

rng = np.random.default_rng(a.seed)
tr, va, te = make("train", rng, True), make("val", rng, False), make("test", rng, False)

def loader(d, shuffle):
    X, P, D, Y = d[:4]
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(P),
                       torch.from_numpy(np.repeat(D[:, None, :], W, 1)), torch.from_numpy(Y.astype(np.float32)))
    return DataLoader(ds, batch_size=a.batch, shuffle=shuffle, num_workers=0)
tl, vl, el = loader(tr, True), loader(va, False), loader(te, False)
DELTA = torch.arange(W, 0, -1).float()                      # [W,...,1]  (v1:53)

# ── 모델 (v1:174-289 와 동일 구조, attention 만 시점 마스크) ─────
class MaskingLayer(nn.Module):
    def forward(self, x):
        m = (x != MASK).float(); return x * m, m
class WeightedInputLayer(nn.Module):
    def __init__(self, nf):
        super().__init__(); self.alpha = nn.Parameter(torch.ones(nf) * 0.1); self.beta = nn.Parameter(torch.ones(nf))
    def forward(self, x, dt): return x * (self.beta * torch.exp(-self.alpha * dt.unsqueeze(-1)))
class AttentionLayer(nn.Module):
    def __init__(self, h): super().__init__(); self.attn = nn.Linear(h, 1)
    def forward(self, o, pad):
        s = self.attn(o).masked_fill(pad.unsqueeze(-1), float("-inf"))
        w = F.softmax(s, dim=1); return (o * w).sum(1), w
class VitalAgent(nn.Module):
    def __init__(self, nv, nd=2, hidden=128, dropout=0.3):
        super().__init__()
        self.masking = MaskingLayer(); self.weighted_input = WeightedInputLayer(nv)
        self.lstm = nn.LSTM(2 * nv + nd, hidden, 2, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout); self.attention = AttentionLayer(hidden)
        self.dense = nn.Linear(hidden, hidden // 2); self.dense_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden // 2, 1)
    def forward(self, vx, pad, dx, dt):
        vm, mask = self.masking(vx); vw = self.weighted_input(vm, dt)
        o, _ = self.lstm(torch.cat([vw, mask, dx], -1)); o = self.lstm_dropout(o)
        ctx, attn = self.attention(o, pad)
        h = self.dense_dropout(F.relu(self.dense(ctx)))
        logits = self.output(h)
        return {"logits": logits, "sepsis_prob": torch.sigmoid(logits), "attn_weights": attn, "context": ctx}

model = VitalAgent(V).to(DEV)
pw = torch.tensor([a.neg_ratio], device=DEV)
def focal(logits, y):                                        # v2:64-71
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
    p = torch.sigmoid(logits); pt = p * y + (1 - p) * (1 - y)
    return ((1 - pt) ** a.gamma * bce).mean()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

@torch.no_grad()
def predict(dl):
    model.eval(); out = []
    for vx, pad, dx, y in dl:
        dt = DELTA.expand(len(vx), -1).to(DEV)
        out.append(model(vx.to(DEV), pad.to(DEV), dx.to(DEV), dt)["sepsis_prob"].squeeze(-1).cpu().numpy())
    return np.concatenate(out)
def metrics(y, p): return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}

# ── 학습 ────────────────────────────────────────────────────────
best, bad, hist = -1, 0, []
for ep in range(1, a.epochs + 1):
    model.train(); tot = 0
    for vx, pad, dx, y in tl:
        dt = DELTA.expand(len(vx), -1).to(DEV)
        loss = focal(model(vx.to(DEV), pad.to(DEV), dx.to(DEV), dt)["logits"].squeeze(-1), y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
    sch.step()
    vm = metrics(va[3], predict(vl)); hist.append({"epoch": ep, "loss": tot / len(tr[3]), **{"val_" + k: v for k, v in vm.items()}})
    log(f"ep{ep:02d} loss={tot/len(tr[3]):.4f} val AUROC={vm['auroc']:.4f} AUPRC={vm['auprc']:.4f}")
    if vm["auroc"] > best:
        best, bad = vm["auroc"], 0; torch.save(model.state_dict(), CKPT)
    else:
        bad += 1
        if bad >= a.patience: log("early stop"); break

# ── 평가 + 저장 ─────────────────────────────────────────────────
model.load_state_dict(torch.load(CKPT, map_location=DEV))
pte = predict(el); yte, hte, hrte, pte_pad = te[3], te[4], te[5], te[1].any(1)
res = {
    "run_name": a.run_name, "tag": a.tag, "window": W, "features": FEATURES, "seed": a.seed, "frac": a.frac,
    "args": vars(a), "device": DEV, "best_val_auroc": best, "best_epoch": int(np.argmax([h["val_auroc"] for h in hist]) + 1),
    "n": {s: {"N": int(len(d[3])), "pos": int(d[3].sum())} for s, d in zip(("train", "val", "test"), (tr, va, te))},
    "neg_min_hist": a.neg_min_hist,
    "pad_rate": {s: {"pos": float(d[1].any(1)[d[3] == 1].mean()), "neg": float(d[1].any(1)[d[3] == 0].mean())}
                 for s, d in zip(("train", "val", "test"), (tr, va, te))},
    "test": metrics(yte, pte),
    "val": metrics(va[3], predict(vl)),
    "history": hist,
    "alpha": dict(zip(FEATURES, model.weighted_input.alpha.detach().cpu().tolist())),
    "beta":  dict(zip(FEATURES, model.weighted_input.beta.detach().cpu().tolist())),
}
# h 버킷별: 해당 버킷 양성 vs 전체 음성
neg = yte == 0
res["test_by_h_bucket"] = {}
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]:
    sel = (yte == 1) & (hte > lo) & (hte <= hi)
    if sel.sum() >= 10:
        idx = sel | neg
        res["test_by_h_bucket"][f"{lo}<h<={hi}"] = {"n_pos": int(sel.sum()), **metrics(yte[idx], pte[idx])}
# 이력 길이별 (패딩 유무)
res["test_by_padding"] = {}
for name, sel in [("full_history", ~pte_pad), ("padded", pte_pad)]:
    if (yte[sel] == 1).sum() >= 10 and (yte[sel] == 0).sum() >= 10:
        res["test_by_padding"][name] = {"n": int(sel.sum()), "n_pos": int(yte[sel].sum()), **metrics(yte[sel], pte[sel])}
json.dump(res, open(RES, "w"), indent=2, ensure_ascii=False)
np.savez(os.path.join(a.out_dir, "results", f"vital_{a.run_name}_testpred.npz"), prob=pte, label=yte, h=hte, hist_rows=hrte)
log(f"TEST AUROC={res['test']['auroc']:.4f} AUPRC={res['test']['auprc']:.4f} | by_h={ {k: round(v['auroc'],3) for k,v in res['test_by_h_bucket'].items()} }")
log(f"저장: {CKPT}, {RES}")
