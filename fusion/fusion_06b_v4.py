"""
fusion_06b_v4.py — Cross-Attention 융합 (fusion_06b_win4.py 와 모델·학습 동일, 데이터/시드/출력명만 인자화)
입력 : --data 의 {train,val,test}.npz (fusion_data_v3 = 바이탈 v3, fusion_data_v4 = 바이탈 v4)
구조 : vital(128)/emr(768) → proj(d) → [CLS,vital,emr] self-attn → CLS → head
비교 : vital 단독 / emr 단독 / concat / 융합
출력 : models/fusion_<name>_s<seed>/fusion.pt, fusion_06_<name>_s<seed>_metrics.json,
       data/fusion_pred_<name>_s<seed>.npz (test prob + CLS attention + 단독확률 + meta)
실행 : python fusion_06b_v4.py --data ./data/fusion_data_v4 --name v4 --seed 42
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"./data/fusion_data_v4")
ap.add_argument("--name", default="v4")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
RUN = f"{a.name}_s{a.seed}"
torch.manual_seed(a.seed); np.random.seed(a.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = a.data
MDIR = os.path.join(r"./models", f"fusion_{RUN}"); os.makedirs(MDIR, exist_ok=True)

def load(s):
    z = np.load(os.path.join(DATA, f"{s}.npz"), allow_pickle=True)
    return (z["vital_vec"], z["emr_vec"], z["label"],
            z["vital_prob"], z["emr_prob"], z["stay_id"], z["t_point"])
Vtr,Etr,ytr,vptr,eptr,_,_ = load("train")
Vva,Eva,yva,_,_,_,_       = load("val")
Vte,Ete,yte,vpte,epte,ste,tte = load("test")
print(f"train {len(ytr)} / val {len(yva)} / test {len(yte)}  pos%={ytr.mean()*100:.2f}", flush=True)

# ── 단독/concat baseline ────────────────────────────────────
def probe(Xtr,Xva,Xte):
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(sc.transform(Xtr),ytr)
    pt=clf.predict_proba(sc.transform(Xte))[:,1]
    return roc_auc_score(yte,pt),average_precision_score(yte,pt)
v_te,v_ap = probe(Vtr,Vva,Vte)
e_te,e_ap = probe(Etr,Eva,Ete)
c_te,c_ap = probe(np.hstack([Vtr,Etr]),np.hstack([Vva,Eva]),np.hstack([Vte,Ete]))
print(f"[단독확률] vital={roc_auc_score(yte,vpte):.4f} | emr={roc_auc_score(yte,epte):.4f}", flush=True)
print(f"[probe] vital={v_te:.4f} | emr={e_te:.4f} | concat={c_te:.4f} (AUPRC {c_ap:.4f})", flush=True)

# ── 융합 모델 ───────────────────────────────────────────────
class Fusion(nn.Module):
    def __init__(self, dv=128, de=768, d=128, nhead=4, drop=0.1):
        super().__init__()
        self.pv = nn.Linear(dv, d); self.pe = nn.Linear(de, d)   # 입력 차원은 데이터에서 (EMR 모델 교체 대응)
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        self.mha = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
        self.ff = nn.Sequential(nn.Linear(d,256), nn.GELU(), nn.Dropout(drop), nn.Linear(256,d))
        self.head = nn.Linear(d,1)
    def forward(self, v, e, need_attn=False):
        B=v.size(0)
        x = torch.cat([self.cls.expand(B,-1,-1),
                       self.pv(v).unsqueeze(1), self.pe(e).unsqueeze(1)], dim=1)
        a,w = self.mha(x,x,x, need_weights=need_attn, average_attn_weights=True)
        x = self.n1(x + self.drop(a)); x = self.n2(x + self.drop(self.ff(x)))
        return self.head(x[:,0]).squeeze(-1), w

def to_t(*a): return [torch.from_numpy(x).float().to(DEV) for x in a]
Vtr_t,Etr_t = to_t(Vtr,Etr); Vva_t,Eva_t = to_t(Vva,Eva); Vte_t,Ete_t = to_t(Vte,Ete)
ytr_t = torch.from_numpy(ytr).float().to(DEV)

model = Fusion(Vtr.shape[1], Etr.shape[1]).to(DEV)
pw = torch.tensor([min((len(ytr)-ytr.sum())/ytr.sum(), 10.0)], device=DEV)
lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"융합 파라미터 {sum(p.numel() for p in model.parameters()):,} | pos_weight {pw.item():.1f}", flush=True)

@torch.no_grad()
def ev(V,E,y):
    # 6~7만 행을 한 번에 넣으면 native MHA 커널이 CUDA invalid configuration 으로 죽음 → 청크 평가
    model.eval(); p = np.concatenate([torch.sigmoid(model(V[i:i+8192],E[i:i+8192])[0]).cpu().numpy()
                                      for i in range(0,len(V),8192)])
    return roc_auc_score(y,p), average_precision_score(y,p), p

BATCH=256; best=-1; best_state=None; wait=0
N=len(ytr); idx=np.arange(N)
for ep in range(1,41):
    model.train(); np.random.shuffle(idx); tot=0
    for i in range(0,N,BATCH):
        b=idx[i:i+BATCH]
        opt.zero_grad()
        logit,_ = model(Vtr_t[b],Etr_t[b])
        loss = lossf(logit,ytr_t[b]); loss.backward(); opt.step(); tot += loss.item()
    va_auc,va_ap,_ = ev(Vva_t,Eva_t,yva)
    if va_auc > best:
        best=va_auc; wait=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(best_state, os.path.join(MDIR, "fusion.pt")); mark="  *best"
    else:
        wait+=1; mark=""
    if ep%2==0 or mark:
        print(f"ep{ep:2d} loss{tot/(N//BATCH+1):.4f} val AUROC={va_auc:.4f} AUPRC={va_ap:.4f}{mark}", flush=True)
    if wait>=7: print("early stop"); break

model.load_state_dict(best_state)
f_te,f_te_ap,p_te = ev(Vte_t,Ete_t,yte)

print("\n===== 융합 결과 (test) =====")
print(f"  vital 단독 : AUROC {v_te:.4f}  AUPRC {v_ap:.4f}")
print(f"  emr   단독 : AUROC {e_te:.4f}  AUPRC {e_ap:.4f}")
print(f"  concat     : AUROC {c_te:.4f}  AUPRC {c_ap:.4f}")
print(f"  융합(X-Attn): AUROC {f_te:.4f}  AUPRC {f_te_ap:.4f}")

json.dump({"vital_only_test_auroc":float(v_te),"vital_only_test_auprc":float(v_ap),
           "emr_only_test_auroc":float(e_te),"emr_only_test_auprc":float(e_ap),
           "concat_test_auroc":float(c_te),"concat_test_auprc":float(c_ap),
           "fusion_test_auroc":float(f_te),"fusion_test_auprc":float(f_te_ap),
           "vital_prob_test_auroc":float(roc_auc_score(yte,vpte)),"emr_prob_test_auroc":float(roc_auc_score(yte,epte)),
           "best_val_auroc":float(best),"data":DATA,"seed":a.seed},
          open(os.path.join(r".", f"fusion_06_{RUN}_metrics.json"),"w"), indent=2)

# XAI: CLS attention(에이전트 의존도) + 단독확률(충돌감지) + meta 저장
model.eval()
with torch.no_grad():
    cls_attn = np.concatenate([model(Vte_t[i:i+8192],Ete_t[i:i+8192], need_attn=True)[1][:,0,:].cpu().numpy()
                               for i in range(0,len(Vte_t),8192)])   # CLS→[CLS,vital,emr]
np.savez_compressed(os.path.join(r"./data", f"fusion_pred_{RUN}.npz"),
    test_prob=p_te, test_y=yte, vital_prob=vpte, emr_prob=epte,
    cls_attn=cls_attn, stay_id=ste, t_point=tte)
print(f"\n=== 6b 완료: {MDIR}, fusion_06_{RUN}_metrics.json, fusion_pred_{RUN}.npz ===")
