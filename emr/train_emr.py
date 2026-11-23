"""
EMR 에이전트 범용 학습 스크립트 : 인코더(BERT 계열) / 디코더(LLM) 모두 LoRA 로 이진 분류
  - 입력: EMR 스냅샷 문서 parquet (stay_id, t_point, label, split, emr_text)
  - 인코더: mean-pooling + Linear / 디코더: 문서 끝 EOS 의 마지막 토큰 + Linear
  - epoch 마다 val AUROC 최고면 저장, patience 회 연속 개선 없으면 조기 종료, best 가중치 복원 후 val/test 평가
  - 결과: <root>/results/<run>.json, <root>/results/<run>.md (공유용 요약), <root>/preds/<run>.npz, <root>/models/<run>/
  - v2 (share_fullft_posw): --pos_weight (양성 가중치 직접 지정), --eval_every (epoch 중간 val 평가) 추가.
    두 옵션 모두 기본값 0 이면 v1 과 완전히 같은 동작 (학습·평가 로직 동일).

예) 인코더 : python train_emr.py --docs emr_docs_mini_v3_L48.parquet --model answerdotai/ModernBERT-base --run modernbert_mini
    LLM    : python train_emr.py --docs emr_docs_mini_v3_L48.parquet --model Qwen/Qwen3-1.7B --run qwen3_1p7b_mini
    점검   : 위 명령 뒤에 --smoke 48 --epochs 1
"""
import os, sys, time, json, argparse, platform, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch, torch.nn as nn
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True, help="문서 parquet")
ap.add_argument("--model", required=True, help="HF 모델 id 또는 로컬 폴더")
ap.add_argument("--run", required=True, help="실행 이름 (결과 파일 이름)")
ap.add_argument("--root", default=".", help="출력 기준 폴더")
ap.add_argument("--arch", default="auto", choices=["auto", "encoder", "decoder"])
ap.add_argument("--maxlen", type=int, default=0, help="0 이면 인코더 512 / 디코더 1280 (모델 최대 길이로 제한)")
ap.add_argument("--batch", type=int, default=0, help="0 이면 인코더 16 / 디코더 8")
ap.add_argument("--grad_accum", type=int, default=1, help="VRAM 부족 시 batch 를 줄이고 이 값을 올림")
ap.add_argument("--eval_batch", type=int, default=0, help="0 이면 batch*2")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--lr", type=float, default=0, help="0 이면 인코더 2e-4 / 디코더 1e-4")
ap.add_argument("--r", type=int, default=16, help="LoRA rank")
ap.add_argument("--lora_targets", default="auto", help="auto | all-linear | 쉼표 구분 모듈 이름")
ap.add_argument("--full_ft", action="store_true", help="LoRA 대신 전체 미세조정 (작은 인코더만, lr 2e-5 권장)")
ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
ap.add_argument("--grad_ckpt", default="auto", choices=["auto", "on", "off"], help="auto: 디코더만 on")
ap.add_argument("--train_neg_ratio", type=float, default=0, help="train 음성:양성 다운샘플 (0=안 함). 큰 데이터용")
ap.add_argument("--val_neg_ratio", type=float, default=0, help="val 음성:양성 다운샘플 (0=안 함). 큰 데이터용")
ap.add_argument("--pos_weight", type=float, default=0, help="BCE 양성 가중치. 0 이면 자동 (train 음성/양성)")
ap.add_argument("--eval_every", type=int, default=0, help=">0 이면 이 배치 수마다 epoch 중간에도 val 평가·best 저장 "
                "(patience 는 평가 횟수 기준). 0 이면 epoch 끝에만")
ap.add_argument("--save_vec", action="store_true", help="pooled 벡터도 저장 (융합 단계 입력용, 용량 큼)")
ap.add_argument("--trust_remote_code", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke", type=int, default=0, help=">0 이면 split 당 이 개수만 써서 전체 흐름 점검")
ap.add_argument("--export_from", default="", help="학습된 models/<run> 폴더. 주면 학습 없이 train/val/test 전체(다운샘플 없음)의 "
                "확률+벡터를 preds/<run>_vec_<split>.npz 로 저장 (융합 입력용). --model/--r/--lora_targets/--maxlen/--full_ft 는 학습 때와 같게")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
assert torch.cuda.is_available(), "CUDA GPU 가 필요합니다"
dev = "cuda"
log = lambda *x: print(time.strftime("[%H:%M:%S]"), *x, flush=True)
OUT_MODEL = os.path.join(a.root, "models", a.run)
RES_DIR, PRED_DIR = os.path.join(a.root, "results"), os.path.join(a.root, "preds")
for d in (OUT_MODEL, RES_DIR, PRED_DIR): os.makedirs(d, exist_ok=True)

# ── 구조 판별 ───────────────────────────────────────────────
cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
if a.arch == "auto":
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    is_dec = ("CausalLM" in archs) or ("ForConditionalGeneration" in archs and not cfg.is_encoder_decoder)
    a.arch = "decoder" if is_dec else "encoder"
assert not getattr(cfg, "is_encoder_decoder", False), "T5 같은 인코더-디코더는 지원 안 함 (--arch 로 강제하지 말 것)"
DEC = a.arch == "decoder"
max_pos = getattr(cfg, "max_position_embeddings", None) or 512
a.maxlen = min(a.maxlen or (1280 if DEC else 512), max_pos)
a.batch = a.batch or (8 if DEC else 16)
a.eval_batch = a.eval_batch or a.batch * 2
a.lr = a.lr or (1e-4 if DEC else (2e-5 if a.full_ft else 2e-4))
if a.precision == "auto":
    a.precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
if DEC and a.precision != "bf16":
    log("경고: 이 GPU 는 bf16 미지원 → LLM 은 fp16 가중치로 로드 (불안정할 수 있음, 인코더 모델 권장)")
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.precision]
gpu = torch.cuda.get_device_properties(0)
log(f"run={a.run} model={a.model} arch={a.arch} maxlen={a.maxlen} batch={a.batch}x{a.grad_accum} lr={a.lr} "
    f"precision={a.precision} GPU={gpu.name} {gpu.total_memory/2**30:.1f}GB")

# ── 데이터 ──────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
EOS = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else (EOS if EOS is not None else 0)
df = pq.read_table(a.docs, columns=["stay_id", "t_point", "label", "split", "emr_text"]).to_pandas()
rng = np.random.default_rng(a.seed)

def subsample(sub, ratio):
    y = sub.label.to_numpy(); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    neg = rng.choice(neg, min(len(neg), int(len(pos) * ratio)), replace=False)
    return sub.iloc[np.sort(np.concatenate([pos, neg]))].reset_index(drop=True)

data = {}
for s, ratio in [("train", a.train_neg_ratio), ("val", a.val_neg_ratio), ("test", 0)]:
    sub = df[df.split == s].reset_index(drop=True)
    if ratio and not a.export_from: sub = subsample(sub, ratio)
    if a.smoke: sub = sub.sample(min(a.smoke, len(sub)), random_state=a.seed).reset_index(drop=True)
    texts = sub.emr_text.tolist()
    if DEC:   # 디코더: 특수토큰 없이 자르고 끝에 EOS → 마지막 토큰이 전체 문맥을 봄
        ids = tok(texts, truncation=True, max_length=a.maxlen - 1, add_special_tokens=False)["input_ids"]
        ids = [x + [EOS] for x in ids]
    else:
        ids = tok(texts, truncation=True, max_length=a.maxlen, add_special_tokens=True)["input_ids"]
    data[s] = dict(ids=ids, y=sub.label.to_numpy(np.float32), stay_id=sub.stay_id.to_numpy(),
                   t_point=sub.t_point.astype(str).to_numpy(), lens=np.array([len(x) for x in ids]))
    log(f"{s}: N={len(ids):,} pos={int(sub.label.sum()):,} 토큰 중앙={int(np.median(data[s]['lens']))} "
        f"잘림={np.mean(data[s]['lens'] >= a.maxlen)*100:.2f}%")
del df

def collate(idx, d):
    L = max(d["lens"][i] for i in idx)
    x = np.full((len(idx), L), PAD, np.int64); m = np.zeros((len(idx), L), np.int64)
    for k, i in enumerate(idx):
        n = d["lens"][i]; x[k, :n] = d["ids"][i]; m[k, :n] = 1      # 오른쪽 패딩
    return {"input_ids": torch.from_numpy(x), "attention_mask": torch.from_numpy(m)}, torch.from_numpy(d["y"][idx])

def batches(d, shuffle, bs):
    n = len(d["y"])
    if not shuffle:
        order = np.argsort(d["lens"]); return [order[i:i + bs] for i in range(0, n, bs)]
    perm = rng.permutation(n); CH = bs * 100; out = []
    for c in range(0, n, CH):            # 청크 안 길이순 → 배치 → 배치 순서 셔플 (패딩 낭비 감소)
        ch = perm[c:c + CH]; ch = ch[np.argsort(d["lens"][ch])]
        out += [ch[i:i + bs] for i in range(0, len(ch), bs)]
    rng.shuffle(out); return out

n_pos = int(data["train"]["y"].sum()); n_neg = len(data["train"]["y"]) - n_pos
pos_weight = torch.tensor([a.pos_weight or n_neg / max(n_pos, 1)], device=dev)
log(f"pos_weight={pos_weight.item():.2f} ({'지정' if a.pos_weight else '자동'}, train 음성/양성 = {n_neg / max(n_pos, 1):.2f})")

# ── 모델 ────────────────────────────────────────────────────
def lora_wrap(base):
    if a.lora_targets == "auto":
        try:
            return get_peft_model(base, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none"))
        except ValueError:      # peft 기본 매핑에 없는 모델 → 모든 선형층
            log("LoRA 기본 대상 없음 → all-linear")
            targets = "all-linear"
    else:
        targets = a.lora_targets if a.lora_targets == "all-linear" else [t.strip() for t in a.lora_targets.split(",")]
    return get_peft_model(base, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", target_modules=targets))

class EmrModel(nn.Module):
    def __init__(self):
        super().__init__()
        load_dtype = AMP if (DEC and AMP is not None) else torch.float32   # 인코더는 fp32 가중치 + autocast
        base = AutoModel.from_pretrained(a.model, dtype=load_dtype, trust_remote_code=a.trust_remote_code)
        if a.grad_ckpt == "on" or (a.grad_ckpt == "auto" and DEC):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(base.config, "use_cache"): base.config.use_cache = False
            base.enable_input_require_grads()
        self.enc = base if a.full_ft else lora_wrap(base)
        H = base.config.hidden_size
        self.drop = nn.Dropout(0.1); self.head = nn.Linear(H, 1)    # head 는 fp32
    def forward(self, enc):
        h = self.enc(**enc).last_hidden_state
        m = enc["attention_mask"]
        if DEC:
            pooled = h[torch.arange(h.size(0), device=h.device), m.sum(1) - 1].float()
        else:
            mf = m.unsqueeze(-1).to(h.dtype); pooled = ((h * mf).sum(1) / mf.sum(1).clamp(min=1)).float()
        return self.head(self.drop(pooled)).squeeze(-1), pooled

model = EmrModel().to(dev); model.train()
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
log(f"전체 파라미터 {n_all/1e6:.1f}M, 학습 {n_tr:,} ({n_tr / n_all * 100:.3f}%)")

lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=(a.precision == "fp16"))
steps_per_ep = int(np.ceil(len(data["train"]["y"]) / a.batch / a.grad_accum)) + 1
sched = get_linear_schedule_with_warmup(opt, max(1, int(steps_per_ep * a.epochs * 0.03)), steps_per_ep * a.epochs)

def autocast():
    return torch.autocast(dev, dtype=AMP) if AMP is not None else torch.autocast(dev, enabled=False)

@torch.no_grad()
def predict(s, want_vec=False):
    model.eval(); d = data[s]; n = len(d["y"])
    P = np.zeros(n, np.float32); V = np.zeros((n, model.head.in_features), np.float16) if want_vec else None
    for idx in batches(d, False, a.eval_batch):
        enc, _ = collate(idx, d); enc = {k: v.to(dev) for k, v in enc.items()}
        with autocast():
            logit, pooled = model(enc)
        P[idx] = torch.sigmoid(logit.float()).cpu().numpy()
        if want_vec: V[idx] = pooled.cpu().numpy().astype(np.float16)
    model.train(); return P, V

# ── 융합용 벡터 내보내기 (학습 없음) ─────────────────────────
if a.export_from:
    if a.full_ft:
        model.load_state_dict(torch.load(os.path.join(a.export_from, "full_ft_trainable.pt"), map_location=dev), strict=False)
    else:
        from peft import load_peft_weights, set_peft_model_state_dict
        r = set_peft_model_state_dict(model.enc, load_peft_weights(a.export_from, device=dev))
        assert not getattr(r, "unexpected_keys", []), "어댑터 키 불일치 → --model/--r/--lora_targets 를 학습 때와 같게"
    model.head.load_state_dict(torch.load(os.path.join(a.export_from, "head.pt"), map_location=dev))
    log(f"export: {a.export_from} 가중치 로드")
    for s in ("train", "val", "test"):
        P, V = predict(s, True); d = data[s]
        np.savez_compressed(os.path.join(PRED_DIR, f"{a.run}_vec_{s}.npz"), prob=P, vec=V, label=d["y"].astype(np.int64),
                            stay_id=d["stay_id"], t_point=np.asarray(d["t_point"], dtype=str))
        log(f"export {s}: N={len(P):,} AUROC={roc_auc_score(d['y'], P):.4f} → {a.run}_vec_{s}.npz")
    sys.exit(0)

# ── 학습 ────────────────────────────────────────────────────
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
best_auc, best_ep, wait, best_state, hist = -1, -1, 0, None, []

def evaluate(ep_tag, loss_avg, t0):
    """val 평가 → best 면 저장. 조기 종료해야 하면 True"""
    global best_auc, best_ep, wait, best_state
    Pv, _ = predict("val")
    va_auc = float(roc_auc_score(data["val"]["y"], Pv)); va_ap = float(average_precision_score(data["val"]["y"], Pv))
    hist.append({"epoch": ep_tag, "loss": loss_avg, "val_auroc": va_auc, "val_auprc": va_ap, "min": (time.time() - t0) / 60})
    log(f"[epoch {ep_tag}] {(time.time()-t0)/60:.1f}분 loss={loss_avg:.4f} val AUROC={va_auc:.4f} AUPRC={va_ap:.4f}")
    if va_auc > best_auc:
        best_auc, best_ep, wait = va_auc, ep_tag, 0
        sd = model.state_dict(); best_state = {n: sd[n].detach().cpu().clone() for n in trainable}
        if a.full_ft: torch.save(best_state, os.path.join(OUT_MODEL, "full_ft_trainable.pt"))
        else: model.enc.save_pretrained(OUT_MODEL)
        torch.save(model.head.state_dict(), os.path.join(OUT_MODEL, "head.pt"))
        json.dump({"best_epoch": ep_tag, "val_auroc": va_auc, "history": hist}, open(os.path.join(OUT_MODEL, "best.json"), "w"), indent=2)
        log(f"  [best] epoch {ep_tag} 저장 → {OUT_MODEL}")
        return False
    wait += 1; log(f"  개선 없음 ({wait}/{a.patience})")
    if wait >= a.patience: log("early stop"); return True
    return False

torch.cuda.reset_peak_memory_stats(); t_start = time.time(); stop = False
for ep in range(1, a.epochs + 1):
    t0 = time.time(); run = 0.0; seen = 0
    bl = batches(data["train"], True, a.batch)
    for bi, idx in enumerate(bl):
        enc, y = collate(idx, data["train"]); enc = {k: v.to(dev) for k, v in enc.items()}; y = y.to(dev)
        with autocast():
            logit, _ = model(enc)
        loss = lossf(logit.float(), y); scaler.scale(loss / a.grad_accum).backward()
        if (bi + 1) % a.grad_accum == 0 or bi + 1 == len(bl):
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        run += loss.item(); seen += len(idx)
        if bi % 200 == 0:
            el = time.time() - t0; rate = seen / el if el > 0 else 0
            eta = (len(data["train"]["y"]) - seen) / rate / 60 if rate > 0 else 0
            log(f"  ep{ep} step{bi}/{len(bl)} loss{run/(bi+1):.4f} {rate:.1f}docs/s 남은시간 {eta:.0f}분")
        if a.eval_every and (bi + 1) % a.eval_every == 0 and bi + 1 < len(bl):     # epoch 중간 평가 (끝은 아래에서)
            if not np.isfinite(run): break
            if evaluate(round(ep - 1 + (bi + 1) / len(bl), 2), run / (bi + 1), t0): stop = True; break
    if not np.isfinite(run):
        log("loss 가 NaN/inf → 발산. --lr 을 절반으로 줄여 다시 실행하세요"); break
    if stop or evaluate(ep, run / len(bl), t0): break
train_min = (time.time() - t_start) / 60

# ── 최종 평가 (best 복원) ───────────────────────────────────
assert best_state is not None, "best 모델이 없습니다 (첫 epoch 부터 발산)"
sd = model.state_dict(); sd.update({n: best_state[n].to(dev) for n in best_state}); model.load_state_dict(sd)
log(f"best epoch {best_ep} 가중치 복원 → val/test 평가")
Pv, Vv = predict("val", a.save_vec); Pt, Vt = predict("test", a.save_vec)
yv, yt = data["val"]["y"], data["test"]["y"]

def metrics(y, p, thr):
    pred = p >= thr; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    fpr, tpr, _ = roc_curve(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "threshold": float(thr),
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "f1": 2 * ppv * sens / max(ppv + sens, 1e-9),
            "sens_at_spec90": float(tpr[fpr <= 0.10].max()) if (fpr <= 0.10).any() else 0.0,
            "prevalence": float(y.mean())}
pr, rc, th = precision_recall_curve(yv, Pv)          # 임계값은 val 에서 F1 최대로 정해 test 에 그대로 적용
f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None); thr = float(th[np.argmax(f1[:-1])])
mv, mt = metrics(yv, Pv, thr), metrics(yt, Pt, thr)
log(f"=== 최종 (best epoch {best_ep}) val AUROC={mv['auroc']:.4f} AUPRC={mv['auprc']:.4f} | "
    f"test AUROC={mt['auroc']:.4f} AUPRC={mt['auprc']:.4f} sens={mt['sensitivity']:.3f} spec={mt['specificity']:.3f}")

np.savez_compressed(os.path.join(PRED_DIR, f"{a.run}.npz"),
    val_prob=Pv, val_y=yv, val_stay_id=data["val"]["stay_id"], val_t_point=data["val"]["t_point"],
    test_prob=Pt, test_y=yt, test_stay_id=data["test"]["stay_id"], test_t_point=data["test"]["t_point"],
    **({"val_vec": Vv, "test_vec": Vt} if a.save_vec else {}))
env = {"gpu": gpu.name, "vram_gb": round(gpu.total_memory / 2**30, 1),
       "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "torch": torch.__version__, "os": platform.platform()}
import transformers, peft; env.update(transformers=transformers.__version__, peft=peft.__version__)
res = {"run": a.run, "model": a.model, "docs": os.path.basename(a.docs), "args": vars(a),
       "n_params_M": round(n_all / 1e6, 1), "n_trainable": n_tr, "pos_weight_used": round(pos_weight.item(), 4),
       "best_epoch": best_ep, "train_minutes": round(train_min, 1),
       "history": hist, "n": {s: {"N": int(len(data[s]["y"])), "pos": int(data[s]["y"].sum()),
                                  "trunc_pct": round(float(np.mean(data[s]["lens"] >= a.maxlen) * 100), 2)} for s in data},
       "val": mv, "test": mt, "env": env}
json.dump(res, open(os.path.join(RES_DIR, f"{a.run}.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# 공유용 요약 md (숫자만, 환자/stay ID 없음)
rows = "\n".join(f"| {h['epoch']} | {h['loss']:.4f} | {h['val_auroc']:.4f} | {h['val_auprc']:.4f} | {h['min']:.0f} |" for h in hist)
md = f"""# EMR 학습 결과: {a.run}

| 항목 | 값 |
|---|---|
| 모델 | `{a.model}` ({a.arch}, {n_all/1e6:.0f}M, {'전체 미세조정' if a.full_ft else f'LoRA r={a.r}'}) |
| 데이터 | `{os.path.basename(a.docs)}` — train {res['n']['train']['N']:,} / val {res['n']['val']['N']:,} / test {res['n']['test']['N']:,} |
| 설정 | maxlen {a.maxlen} (잘림 train {res['n']['train']['trunc_pct']}%), batch {a.batch}x{a.grad_accum}, lr {a.lr}, {a.precision}, pos_weight {pos_weight.item():.2f}{f', eval_every {a.eval_every}' if a.eval_every else ''} |
| best epoch | {best_ep} / 학습 {train_min:.0f}분 |
| GPU | {env['gpu']} ({env['vram_gb']}GB, 최대 사용 {env['peak_vram_gb']}GB) |

## 성능 (best epoch 복원 후)
| split | AUROC | AUPRC | Brier | 민감도 | 특이도 | PPV | F1 | 특이도90% 민감도 |
|---|---|---|---|---|---|---|---|---|
| val | {mv['auroc']:.4f} | {mv['auprc']:.4f} | {mv['brier']:.4f} | {mv['sensitivity']:.3f} | {mv['specificity']:.3f} | {mv['ppv']:.3f} | {mv['f1']:.3f} | {mv['sens_at_spec90']:.3f} |
| **test** | **{mt['auroc']:.4f}** | **{mt['auprc']:.4f}** | {mt['brier']:.4f} | {mt['sensitivity']:.3f} | {mt['specificity']:.3f} | {mt['ppv']:.3f} | {mt['f1']:.3f} | {mt['sens_at_spec90']:.3f} |

임계값 {thr:.3f} (val F1 최대) 를 test 에 그대로 적용. test 양성 비율 {mt['prevalence']*100:.1f}%.

## epoch 기록
| epoch | train loss | val AUROC | val AUPRC | 분 |
|---|---|---|---|---|
{rows}
"""
open(os.path.join(RES_DIR, f"{a.run}.md"), "w", encoding="utf-8").write(md)
log(f"저장: {RES_DIR}\\{a.run}.json / .md, {PRED_DIR}\\{a.run}.npz, {OUT_MODEL}")
