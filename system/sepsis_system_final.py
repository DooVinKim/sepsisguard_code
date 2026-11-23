"""
sepsis_system_final.py — SepsisGuard 최종 엔진 (바이탈 v3 + EMR PubMedBERT 전체 미세조정 + 3토큰 융합)

  입력 : 가상 환자 JSON (입력 형식 v2, DASHBOARD_RECOMMENDATION.md 4절) + 시점 T (입실 후 경과 시간, 정수)
  출력 : Result dict (DASHBOARD_RECOMMENDATION.md 13-7절의 필드)

  학습과 같은 전처리를 그대로 옮김
    - 바이탈 : 02c_vital_label_split.R  (시간 칸 = floor(t), 칸 안 평균, SI = HR/SBP, train Z-score, 결측 -1)
               fusion_06a_v3.py         (T 이하 측정 칸 중 직전 24 행, 앞쪽 부족분 pad, delta_t = 24..1, age_z)
    - EMR    : 04 / 04c_emr_windowing.R (lab 48h·결과 저장 ≤ T, 배양 채취 24h, 배양 결과 72h, 영상 72h·판독 저장 ≤ T)
               05c_emr_serialize.R      (문서 템플릿·숫자 형식·추세·정렬을 글자 단위로 동일하게)
    - 융합   : fusion_06b_v4.py         ([CLS, vital, emr] 3토큰 self-attention)

  사용 예
    from sepsis_system_final import SepsisEngine
    eng = SepsisEngine()                         # 모델 3개 로드 (CPU 가능)
    r = eng.assess(patient, T=48)                # 한 시점
    traj = eng.trajectory(patient)               # timeline 전 시점 (섹션 기여도는 마지막 시점만)
"""
from __future__ import annotations
import os, re, json, math
from pathlib import Path
from typing import Any

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG = json.load(open(HERE / "config" / "engine_config.json", encoding="utf-8"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════════
# 모델 정의 (학습 코드와 동일)
# ════════════════════════════════════════════════════════════════════
MASK = CFG["vital"]["mask"]


class _Masking(nn.Module):
    def forward(self, x):
        m = (x != MASK).float(); return x * m, m


class _Weighted(nn.Module):
    def __init__(self, nf):
        super().__init__(); self.alpha = nn.Parameter(torch.ones(nf) * 0.1); self.beta = nn.Parameter(torch.ones(nf))
    def forward(self, x, dt): return x * (self.beta * torch.exp(-self.alpha * dt.unsqueeze(-1)))


class _Attn(nn.Module):
    def __init__(self, h): super().__init__(); self.attn = nn.Linear(h, 1)
    def forward(self, o, pad):
        s = self.attn(o).masked_fill(pad.unsqueeze(-1), float("-inf"))
        w = F.softmax(s, dim=1); return (o * w).sum(1), w


class VitalAgent(nn.Module):                      # code/vital_v3/vital_v3_model.py
    def __init__(self, nv, nd=2, hidden=128, dropout=0.3):
        super().__init__()
        self.masking = _Masking(); self.weighted_input = _Weighted(nv)
        self.lstm = nn.LSTM(2 * nv + nd, hidden, 2, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout); self.attention = _Attn(hidden)
        self.dense = nn.Linear(hidden, hidden // 2); self.dense_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden // 2, 1)
    def forward(self, vx, pad, dx, dt):
        vm, mask = self.masking(vx); vw = self.weighted_input(vm, dt)
        o, _ = self.lstm(torch.cat([vw, mask, dx], -1)); o = self.lstm_dropout(o)
        ctx, attn = self.attention(o, pad)
        logits = self.output(self.dense_dropout(F.relu(self.dense(ctx))))
        return torch.sigmoid(logits).squeeze(-1), ctx, attn.squeeze(-1)


class EmrModel(nn.Module):                        # train_emr.py (인코더 + mean-pool + Linear)
    def __init__(self, base):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(base)
        self.drop = nn.Dropout(0.1); self.head = nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype); pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled)).squeeze(-1), pooled


class Fusion(nn.Module):                          # fusion_06b_v4.py
    def __init__(self, dv=128, de=768, d=128, nhead=4, drop=0.1):
        super().__init__()
        self.pv = nn.Linear(dv, d); self.pe = nn.Linear(de, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
        self.ff = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, d))
        self.head = nn.Linear(d, 1)
    def forward(self, v, e):
        B = v.size(0)
        x = torch.cat([self.cls.expand(B, -1, -1), self.pv(v).unsqueeze(1), self.pe(e).unsqueeze(1)], dim=1)
        a, w = self.mha(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.n1(x + self.drop(a)); x = self.n2(x + self.drop(self.ff(x)))
        return torch.sigmoid(self.head(x[:, 0]).squeeze(-1)), w[:, 0, :]    # CLS → [cls, vital, emr]


# ════════════════════════════════════════════════════════════════════
# EMR 문서 직렬화 (05c_emr_serialize.R 과 글자 단위로 동일)
# ════════════════════════════════════════════════════════════════════
D = CFG["emr_doc"]
NO_LABS = f"No relevant labs measured in last {D['win_lab']}h."
NO_DRAWN = f"No cultures drawn in last {D['win_micro']}h."
NO_RES = f"No cultures collected in last {D['win_micro_res']}h."
NO_RADIO = f"No imaging in last {D['win_radio']}h."


def r_round(x: float) -> int:
    """R 의 round() 와 같은 짝수 반올림 (Python round 와 동일)"""
    return int(round(float(x)))


def fmt_num(x: float) -> str:
    """R: trimws(formatC(x, format = 'g', digits = 3))"""
    return f"{float(x):.3g}"


def fmt_h(h: float) -> str:
    return "-%dh" % r_round(h)


def lab_line(a: dict) -> str:
    """a: lab_name, n_meas, vals[], hrs_ago[], uom, ref_low, ref_high, last_val, delta_val, any_abnormal"""
    lo, hi = a.get("ref_low"), a.get("ref_high")
    has_ref = lo is not None and hi is not None and not (isinstance(lo, float) and math.isnan(lo)) \
        and not (isinstance(hi, float) and math.isnan(hi))
    if has_ref:
        status = "HIGH" if a["last_val"] > hi else ("LOW" if a["last_val"] < lo else "normal")
        tol = (hi - lo) * D["trend_tol"] if hi > lo else 0.0
    else:
        status = "abnormal" if a.get("any_abnormal") else "normal"
        tol = 0.0
    if a["n_meas"] > 1:
        dv = a["delta_val"]
        trend = ", rising" if dv > tol else (", falling" if dv < -tol else ", stable")
    else:
        trend = ""
    refpart = f", ref {fmt_num(lo)}-{fmt_num(hi)}" if has_ref else ""
    seq = " → ".join(f"{fmt_num(v)} [{fmt_h(h)}]" for v, h in zip(a["vals"], a["hrs_ago"]))
    uom = a.get("uom") or ""
    line = "- %s (n=%d): %s %s (%s%s%s)" % (a["lab_name"], a["n_meas"], seq, uom, status, refpart, trend)
    return re.sub(r"  +", " ", line)


def clean_comment(x: str) -> str:
    x = re.sub(r"_{2,}", "[redacted]", x); x = re.sub(r"[ \t\r\n]+", " ", x).strip()
    return x[:80] + "..." if len(x) > 80 else x


def clean_text(x: str) -> str:
    x = re.sub(r"_{2,}", "[redacted]", x)
    x = re.sub(r"\[redacted\]( ?\[redacted\])+", "[redacted]", x)
    x = re.sub(r"[=_*#-]{3,}", " ", x)
    x = re.sub(r"[ \t]*\n[ \t]*", " ", x)
    x = re.sub(r"[ \t]{2,}", " ", x)
    return x.strip()


def extract_impression(txt: str) -> str:
    m = re.search(r"(?si)IMPRESSION:\s*(.*)", txt)
    body = re.sub(r"(?si)^IMPRESSION:\s*", "", m.group(0)) if m else txt
    body = clean_text(body)
    return body[:D["imp_max"]] + " ..." if len(body) > D["imp_max"] else body


def result_line(r: dict) -> str:
    """r: spec_type_desc, test_name, c_ago, r_ago, status, organisms, resistant, comment"""
    if r["status"] == "pending":
        return "- %s (%s), collected %dh ago: pending" % (r["spec_type_desc"], r["test_name"], r_round(r["c_ago"]))
    if r.get("organisms"):
        res = r["organisms"] + ("" if not r.get("resistant") else "; resistant: " + r["resistant"])
    elif r.get("comment"):
        res = clean_comment(r["comment"])
    else:
        res = "resulted, no organism reported"
    return "- %s (%s), collected %dh ago, resulted %dh ago: %s" % (
        r["spec_type_desc"], r["test_name"], r_round(r["c_ago"]), r_round(r["r_ago"]), res)


def assemble(labs_block: str | None, drawn_line: str | None, res_block: str | None, radio_block: str | None) -> str:
    return (f"Recent laboratory results (last {D['win_lab']}h):\n{labs_block or NO_LABS}\n\n"
            f"Microbiology (cultures drawn, last {D['win_micro']}h):\n{drawn_line or NO_DRAWN}\n\n"
            f"Culture results (collected in last {D['win_micro_res']}h, results available at this time):\n{res_block or NO_RES}\n\n"
            f"Recent imaging impression (last {D['win_radio']}h):\n{radio_block or NO_RADIO}")


def blocks_from_aggregates(labs: list[dict], drawn: dict | None, results: list[dict], radio: dict | None):
    """윈도잉이 끝난 집계(학습 파이프라인의 중간 산출물과 같은 형태) → 문서 블록 4개"""
    order = {n: i for i, n in enumerate(D["lab_order"])}
    labs = sorted(labs, key=lambda a: order.get(a["lab_name"], 99))
    labs_block = "\n".join(lab_line(a) for a in labs) if labs else None
    drawn_line = None
    if drawn and drawn.get("n_cultures"):
        sp = drawn.get("specimen_types") or ""
        drawn_line = "- Cultures drawn in last %dh: %d. Blood culture: %s. Specimens: %s." % (
            D["win_micro"], drawn["n_cultures"], "yes" if drawn.get("blood_culture") else "no", sp if sp else "n/a")
    res_block = None
    if results:
        rs = sorted(results, key=lambda r: r["c_ago"])
        lines = [result_line(r) for r in rs[:D["max_micro_lines"]]]
        if len(rs) > D["max_micro_lines"]:
            lines.append("(+%d more)" % (len(rs) - D["max_micro_lines"]))
        res_block = "\n".join(lines)
    radio_block = None
    if radio:
        radio_block = '- (%dh ago) "%s"' % (r_round(radio["hrs_ago"]), extract_impression(radio["text"]))
    return labs_block, drawn_line, res_block, radio_block


# ════════════════════════════════════════════════════════════════════
# 엔진
# ════════════════════════════════════════════════════════════════════
class SepsisEngine:
    def __init__(self, root: str | Path = ROOT, device: str = DEV, threads: int = 4, load_emr: bool = True):
        torch.set_num_threads(threads)
        self.root = Path(root); self.dev = device
        P = CFG["paths_relative_to_project_root"]
        V = CFG["vital"]
        self.feat = V["features"]; self.W = V["window_rows"]
        self.norm = json.load(open(self.root / P["vital_norm"], encoding="utf-8"))
        self.vital = VitalAgent(len(self.feat))
        self.vital.load_state_dict(torch.load(self.root / P["vital_ckpt"], map_location="cpu", weights_only=True))
        self.vital.to(device).eval()
        self.fusion = Fusion()
        self.fusion.load_state_dict(torch.load(self.root / P["fusion_ckpt"], map_location="cpu", weights_only=True))
        self.fusion.to(device).eval()
        self.emr = self.tok = None
        if load_emr:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()     # 로딩 진행 막대·사전학습 헤드 안내 숨김
            base = CFG["emr_base_model"]; edir = self.root / P["emr_dir"]
            self.tok = AutoTokenizer.from_pretrained(base)
            self.emr = EmrModel(base)
            sd = torch.load(edir / "full_ft_trainable.pt", map_location="cpu")
            miss = self.emr.load_state_dict(sd, strict=False).missing_keys
            assert not [k for k in miss if k.startswith("enc.")], f"EMR 인코더 가중치 누락 {len(miss)}"
            self.emr.head.load_state_dict(torch.load(edir / "head.pt", map_location="cpu"))
            self.emr.to(device).eval()

    # ── 바이탈 ──────────────────────────────────────────────────────
    def vital_rows(self, patient: dict, T: int):
        """T 이하 측정 칸(시간 칸 = floor(t))별 평균 → 학습과 같은 원값 행 목록 [(bucket, {feat: 값})]"""
        rng = CFG["vital"]["clean_range"]
        buckets: dict[int, dict[str, list]] = {}
        for v in patient.get("vitals", []):
            b = int(math.floor(v["t"]))
            if b > T:
                continue
            rec = buckets.setdefault(b, {})
            temp = v.get("Temp_C")
            if temp is None and v.get("Temp_F") is not None:
                temp = (v["Temp_F"] - 32) * 5 / 9
            for k, val in [("HR", v.get("HR")), ("SBP", v.get("SBP")), ("RR", v.get("RR")), ("SpO2", v.get("SpO2")),
                           ("Temp", temp), ("DBP", v.get("DBP")), ("MAP", v.get("MAP"))]:
                if val is None:
                    continue
                if k in rng and not (rng[k][0] <= val <= rng[k][1]):      # 생리학적 범위 밖 → 결측
                    continue
                rec.setdefault(k, []).append(float(val))
        rows = []
        for b in sorted(buckets):
            r = {k: float(np.mean(vs)) for k, vs in buckets[b].items()}
            r["SI"] = r["HR"] / r["SBP"] if ("HR" in r and "SBP" in r and r["SBP"] != 0) else None
            rows.append((b, r))
        return rows

    def vital_tensors(self, patient: dict, T: int):
        rows = self.vital_rows(patient, T)[-self.W:]
        x = np.full((self.W, len(self.feat)), MASK, np.float32); pad = np.ones(self.W, bool)
        for i, (b, r) in enumerate(rows):
            j = self.W - len(rows) + i; pad[j] = False
            for k, f in enumerate(self.feat):
                val = r.get(f)
                if val is not None:
                    x[j, k] = (val - self.norm[f]["mean"]) / self.norm[f]["sd"]
        demo = patient.get("demo", {})
        age = demo.get("age")
        age_z = 0.0 if age is None else (age - CFG["vital"]["age_mean"]) / CFG["vital"]["age_sd"]
        g = {"M": 1.0, "F": 0.0}.get(demo.get("gender"), 0.5)
        return x, pad, np.array([age_z, g], np.float32), rows

    @torch.no_grad()
    def run_vital(self, X, PAD, DEMO):
        """배치 입력 (B, W, V), (B, W), (B, 2) → 확률, context, attention"""
        vx = torch.as_tensor(X, device=self.dev); pad = torch.as_tensor(PAD, device=self.dev)
        dx = torch.as_tensor(np.repeat(np.asarray(DEMO)[:, None, :], self.W, 1), device=self.dev)
        dt = torch.as_tensor(np.tile(np.arange(self.W, 0, -1, dtype=np.float32), (len(X), 1)), device=self.dev)
        p, ctx, attn = self.vital(vx, pad, dx, dt)
        return p.cpu().numpy(), ctx.cpu().numpy(), attn.cpu().numpy()

    # ── EMR 윈도잉 (04 / 04c) ────────────────────────────────────────
    def emr_aggregates(self, patient: dict, T: int):
        labs = {}
        for l in patient.get("labs", []):
            if T - D["win_lab"] <= l["t_drawn"] <= T and l["t_result"] <= T and l.get("value") is not None:
                labs.setdefault(l["name"], []).append(l)
        lab_aggs = []
        for name, ls in labs.items():
            ls = sorted(ls, key=lambda x: x["t_drawn"])
            vals = [x["value"] for x in ls]; last = ls[-1]
            flags = [x.get("flag") for x in ls]
            any_abn = any(f for f in flags) if any(f is not None for f in flags) else any(
                (x.get("ref_low") is not None and x["value"] < x["ref_low"]) or
                (x.get("ref_high") is not None and x["value"] > x["ref_high"]) for x in ls)
            lab_aggs.append({"lab_name": name, "n_meas": len(ls), "vals": vals,
                             "hrs_ago": [round((T - x["t_drawn"]) * 60) / 60.0 for x in ls],
                             "uom": last.get("unit"), "ref_low": last.get("ref_low"), "ref_high": last.get("ref_high"),
                             "last_val": vals[-1], "delta_val": vals[-1] - vals[0], "any_abnormal": any_abn,
                             "t_last": last["t_drawn"]})
        cul = patient.get("cultures", [])
        d24 = [c for c in cul if T - D["win_micro"] <= c["t_collected"] <= T]
        drawn = None
        if d24:
            specs = []
            for c in d24:
                if c["specimen"] not in specs:
                    specs.append(c["specimen"])
            drawn = {"n_cultures": len(d24), "blood_culture": any("blood" in c["specimen"].lower() for c in d24),
                     "specimen_types": "; ".join(specs)}
        results = []
        for c in cul:
            if not (T - D["win_micro_res"] <= c["t_collected"] <= T):
                continue
            done = c.get("t_resulted") is not None and c["t_resulted"] <= T
            results.append({"spec_type_desc": c["specimen"], "test_name": c["test_name"],
                            "c_ago": T - c["t_collected"], "r_ago": (T - c["t_resulted"]) if done else None,
                            "status": "resulted" if done else "pending",
                            "organisms": c.get("organisms") if done else None,
                            "resistant": c.get("resistant") if done else None,
                            "comment": c.get("comment") if done else None})
        rad = [i for i in patient.get("imaging", []) if T - D["win_radio"] <= i["t_exam"] <= T and i["t_report"] <= T]
        radio = None
        if rad:
            latest = max(rad, key=lambda i: i["t_exam"])
            radio = {"hrs_ago": T - latest["t_exam"], "text": latest["impression"]}
        return lab_aggs, drawn, results, radio

    def emr_document(self, patient: dict, T: int):
        labs, drawn, results, radio = self.emr_aggregates(patient, T)
        blocks = blocks_from_aggregates(labs, drawn, results, radio)
        return assemble(*blocks), blocks, labs, results, radio

    @torch.no_grad()
    def run_emr(self, texts: list[str], batch: int = 8):
        P, V = [], []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], truncation=True, max_length=CFG["emr_maxlen"], padding=True, return_tensors="pt")
            logit, pooled = self.emr(enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev))
            P.append(torch.sigmoid(logit).cpu().numpy()); V.append(pooled.cpu().numpy())
        return np.concatenate(P), np.concatenate(V)

    @torch.no_grad()
    def run_fusion(self, vctx, evec):
        p, w = self.fusion(torch.as_tensor(vctx, device=self.dev, dtype=torch.float32),
                           torch.as_tensor(evec, device=self.dev, dtype=torch.float32))
        return p.cpu().numpy(), w.cpu().numpy()

    # ── 등급·규칙 ────────────────────────────────────────────────────
    @staticmethod
    def tier(score: float, kind: str = "fusion") -> str:
        th = CFG["thresholds"][kind]
        return "HIGH" if score >= th["high"] else ("MODERATE" if score >= th["moderate"] else "LOW")

    @staticmethod
    def guideline(labs: list[dict], rows: list, T: int | None = None, W: int = 6) -> dict:
        G = CFG["guideline_rules"]; last = {a["lab_name"]: a["last_val"] for a in labs}
        flags = []
        for name, (op, thr, meaning) in G["labs"].items():
            if name not in last:
                continue
            v = last[name]
            hit = (op == ">" and v > thr) or (op == "<" and v < thr) or (op == "outside" and (v < thr[0] or v > thr[1]))
            if hit:
                flags.append({"name": name, "value": v, "meaning": meaning + (" (severe)" if name == "Lactate" and v > 4 else "")})
        q = []
        if rows:
            # 각 항목은 바이탈 창(최근 W시간) 안의 마지막 측정값 사용 (T 시점에 측정이 없어도 직전 값 반영)
            t_end = rows[-1][0] if T is None else T
            lb = G["qsofa"].get("lookback_h", W)
            def last_meas(k):
                for b, r in reversed(rows):
                    if b < t_end - lb + 1:
                        break
                    if r.get(k) is not None:
                        return r[k]
                return None
            rr, sbp = last_meas("RR"), last_meas("SBP")
            if rr is not None and rr >= G["qsofa"]["RR"][1]: q.append(f"RR {rr:.0f} ≥ 22")
            if sbp is not None and sbp <= G["qsofa"]["SBP"][1]: q.append(f"SBP {sbp:.0f} ≤ 100")
        strong = len(q) >= 2 or last.get("Lactate", 0) > 4 or len(flags) >= 2
        return {"lab_flags": flags, "qsofa_items": q, "qsofa_count": len(q), "rule_signal": "강함" if strong else ("약함" if (flags or q) else "없음"),
                "note": "규칙 플래그 (lab 5종 · qSOFA 2항목, 의식 상태 미포함). SOFA 점수·감염 의심은 계산하지 않음"}

    # ── 한 시점 판정 ─────────────────────────────────────────────────
    def assess(self, patient: dict, T: int | None = None, sections: bool = True, _cache: dict | None = None) -> dict[str, Any]:
        T = int(patient.get("default_t") if T is None else T)
        x, pad, demo, rows = self.vital_tensors(patient, T)
        if pad.all():
            raise ValueError(f"T={T} 이전 바이탈 기록이 없음")
        vp, vctx, vattn = self.run_vital(x[None], pad[None], demo[None])
        doc, blocks, labs, results, radio = self.emr_document(patient, T)
        ep, evec = self.run_emr([doc])
        fp, w = self.run_fusion(vctx, evec)
        dep_raw = w[0, 1:]; dep = dep_raw / dep_raw.sum()
        f = float(fp[0]); v = float(vp[0]); e = float(ep[0])
        tf, tv, te = self.tier(f), self.tier(v, "vital"), self.tier(e, "emr")
        conflict = {tv, te} == {"HIGH", "LOW"}

        # 섹션 기여도: 해당 섹션을 "No ..." 로 바꿨을 때 EMR 점수 변화 (음수 = 그 섹션이 점수를 올리고 있었음)
        sec = None
        if sections:
            lb, dl, rb, rd = blocks
            alts, names = [], []
            if lb: alts.append(assemble(None, dl, rb, rd)); names.append("labs")
            if dl or rb: alts.append(assemble(lb, None, None, rd)); names.append("cultures")
            if rd: alts.append(assemble(lb, dl, rb, None)); names.append("imaging")
            sec = {"labs": None, "cultures": None, "imaging": None}
            if alts:
                ap, _ = self.run_emr(alts)
                for n, p in zip(names, ap):
                    sec[n] = round(float(p) - e, 4)

        # 바이탈 창 (원값) + attention
        win = []
        nrow = len(rows[-self.W:])
        for i in range(self.W):
            j = i - (self.W - nrow)
            if j < 0:
                win.append(None); continue
            b, r = rows[-self.W:][j]
            win.append({"h": b - T, **{k: (None if r.get(k) is None else round(r[k], 2)) for k in self.feat}})
        abnormal = []
        for a in sorted(labs, key=lambda a: CFG["emr_doc"]["lab_order"].index(a["lab_name"]) if a["lab_name"] in CFG["emr_doc"]["lab_order"] else 99):
            line = lab_line(a)
            m = re.search(r"\((HIGH|LOW)[^)]*?(rising|falling|stable)?\)$", line)
            if m:
                abnormal.append({"name": a["lab_name"], "value": a["last_val"], "unit": a.get("uom"), "flag": m.group(1),
                                 "trend": m.group(2), "hours_ago": round(T - a["t_last"], 1)})

        r = {"patient_id": patient.get("patient_id"), "T": T,
             "fusion_score": round(f, 4), "tier": tf,
             "vital_score": round(v, 4), "emr_score": round(e, 4), "vital_tier": tv, "emr_tier": te, "conflict": conflict,
             "agent_dependency": {"vital": round(float(dep[0]), 4), "emr": round(float(dep[1]), 4)},
             "vital_window": win, "vital_attention": [round(float(a), 4) for a in vattn[0]],
             "vital_row_at_T": bool(rows and rows[-1][0] == T),
             "abnormal_labs": abnormal, "emr_section_effect": sec,
             "cultures": [{k: c[k] for k in ("spec_type_desc", "status", "organisms")} | {"hours_ago": round(c["c_ago"], 1)} for c in results],
             "imaging": None if radio is None else {"hours_ago": round(radio["hrs_ago"], 1)},
             "guideline": self.guideline(labs, rows, T, self.W), "emr_document": doc}
        r["report_md"] = report_md(r, patient)
        return r

    def trajectory(self, patient: dict, sections: str = "last", progress=None) -> dict[int, dict]:
        """timeline_hours 의 모든 시점. sections: 'all' | 'last' (기본: default_t 만) | 'none'"""
        t0, t1 = patient.get("timeline_hours", [patient.get("default_t", 48)] * 2)
        last = int(patient.get("default_t", t1))
        out = {}
        for T in range(int(t0), int(t1) + 1):
            out[T] = self.assess(patient, T, sections=(sections == "all" or (sections == "last" and T == last)))
            if progress: progress(T, out[T])
        return out


# ════════════════════════════════════════════════════════════════════
# 책임의사 리포트 (한국어 마크다운, 규칙 기반 문장)
# ════════════════════════════════════════════════════════════════════
TIER_KO = {"HIGH": "고위험", "MODERATE": "중간위험", "LOW": "저위험"}
ACTION = {"HIGH": "책임의사 즉시 확인 (경보 후 3시간 안 확인 권고)", "MODERATE": "관찰 목록 등록 · 추가 검사 고려", "LOW": "일상 모니터링"}
VNAME = {"HR": "심박수", "SBP": "수축기 혈압", "RR": "호흡수", "SpO2": "산소포화도", "Temp": "체온", "MAP": "평균 동맥압"}


LAB_KO = {"Lactate": "젖산", "WBC": "백혈구", "Neutrophils": "호중구", "Bands": "미성숙 호중구", "Platelet": "혈소판",
          "Creatinine": "크레아티닌", "BUN": "BUN", "Bilirubin": "빌리루빈", "Bicarbonate": "중탄산염", "CRP": "CRP"}
LAB_PRIORITY = ["Lactate", "Platelet", "Creatinine", "Bilirubin", "WBC", "Bands", "CRP", "Neutrophils", "Bicarbonate", "BUN"]
SPEC_KO = {"BLOOD CULTURE": "혈액배양", "URINE": "소변배양", "SPUTUM": "객담배양", "MRSA SCREEN": "MRSA 선별검사"}
ORG_KO = {"ESCHERICHIA COLI": "대장균"}
NARR_MARK = "<!-- details -->"   # 대시보드는 이 표시 앞(서술)만 크게 보여주고 뒤(세부 수치)는 접어 둔다


def _eul(word: str) -> str:   # 받침 여부로 을/를
    c = word[-1]
    return "을" if ("가" <= c <= "힣" and (ord(c) - 0xAC00) % 28) else "를"


def _ro(word: str) -> str:    # 받침(ㄹ 제외) 여부로 으로/로
    c = word[-1]
    if "가" <= c <= "힣":
        jong = (ord(c) - 0xAC00) % 28
        return "으로" if jong and jong != 8 else "로"
    return "로"


def _culture_state(patient: dict, T: float) -> tuple[list[str], list[str], list[str]]:
    """모델 입력과 같은 창(최근 72h 채취)의 배양을 결과 시각까지 포함한 한국어 문구로."""
    pend, done, neg = [], [], []
    for c in patient.get("cultures", []):
        if not (T - D["win_micro_res"] <= c["t_collected"] <= T):
            continue
        spec = c.get("spec_type_desc") or c.get("specimen", "")
        sp = SPEC_KO.get(spec, spec)
        ago = round(T - c["t_collected"])
        if c.get("t_resulted") is None or c["t_resulted"] > T:
            pend.append(f"{sp}({ago}시간 전 채취)")
        elif c.get("organisms"):
            org = ORG_KO.get(c["organisms"], c["organisms"])
            done.append(f"{sp}에서 {org} 검출({round(T - c['t_resulted'])}시간 전 결과)")
        else:
            neg.append(sp)
    return pend, done, neg


RAD_KW_SPECIFIC = {"pneumonia": "폐렴", "consolidation": "폐렴", "infiltrate": "폐렴", "pyelonephritis": "신우신염",
                   "abscess": "농양", "cholecystitis": "담낭염", "cellulitis": "봉와직염", "infection": "감염", "infectious": "감염"}
RAD_KW_NONSPECIFIC = {"opacity": "음영", "effusion": "삼출", "atelectasis": "무기폐", "edema": "부종"}


def radiology_keywords(text: str) -> dict:
    """판독문 키워드 추출 (규칙 기반). 부정 문맥(no/without …)과 호전 문맥(resolution of …)은 양성에서 제외."""
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", (text or "").lower()))
    pos, neg, improved = [], [], False
    for key in list(RAD_KW_SPECIFIC) + list(RAD_KW_NONSPECIFIC):
        hit_pos = hit_neg = False
        for s in sents:
            if not re.search(rf"\b{key}\b", s):
                continue
            if (re.search(rf"\b(resolution of|resolved|resolving|improv\w*)\b[^.]*\b{key}\b", s)
                    or re.search(rf"\b{key}\b[^.]*\b(resolved|resolving|improved)\b", s)):
                improved = True; hit_neg = True
            elif (re.search(rf"\b(no|without|negative for|no evidence of)\b[^.]*\b{key}\b", s)
                    or re.search(rf"\b{key}\b[^.]*\babsent\b", s)):
                hit_neg = True
            else:
                hit_pos = True
        (pos if hit_pos else neg if hit_neg else []).append(key)
    spec = [k for k in pos if k in RAD_KW_SPECIFIC]
    label = None
    if spec:
        ko = RAD_KW_SPECIFIC[spec[0]] if not any(RAD_KW_SPECIFIC[k] == "폐렴" for k in spec) else "폐렴"
        label = f"{ko} 시사 소견"
    return {"label": label, "keys": spec, "nonspecific": [k for k in pos if k in RAD_KW_NONSPECIFIC],
            "negated": neg, "improved": improved}


def narrative_md(r: dict, patient: dict, history: list | None = None) -> list[str]:
    """책임의사가 먼저 읽을 서술형 소견 (규칙 기반 문장 틀).
    요약 → 경과 → 주요 근거 → 에이전트 판단 → Sepsis-3 대조 → 확인 권고 → 판단 한계.
    history: [(t, score0~100, tier), ...] 중 t ≤ T 인 과거 기록만 (미래 점수 사용 금지)."""
    T = r["T"]
    tier, vt, et = r["tier"], r["vital_tier"], r["emr_tier"]
    risk = lambda t: t in ("HIGH", "MODERATE")
    ft = TIER_KO[tier]
    labs_in = patient.get("labs", [])
    has_labs = any(l["t_result"] <= T for l in labs_in)
    ab = {a["name"]: a for a in r["abnormal_labs"]}

    # ── 바이탈 요약 ──
    win = [w for w in r["vital_window"] if w]
    def series(k): return [w[k] for w in win if w.get(k) is not None]
    hr, sbp, rr, spo2, tmp, mp = (series(k) for k in ("HR", "SBP", "RR", "SpO2", "Temp", "MAP"))
    vit_short, vit_long = [], []          # 요약용 짧은 말 / 근거용 수치
    if hr and hr[-1] >= 100:
        vit_short.append("빈맥"); vit_long.append(f"심박수 {hr[0]:g}→{hr[-1]:g}회/분" if hr[-1] - hr[0] >= 10 else f"심박수 {hr[-1]:g}회/분")
    if rr and rr[-1] >= 22:
        vit_short.append("빈호흡"); vit_long.append(f"호흡수 {rr[0]:g}→{rr[-1]:g}회/분" if rr[-1] - rr[0] >= 4 else f"호흡수 {rr[-1]:g}회/분")
    bp_low = bool(sbp and (sbp[-1] <= 100 or sbp[0] - sbp[-1] >= 15))
    if bp_low:
        vit_short.append("혈압 하강"); vit_long.append(f"수축기 혈압 {sbp[0]:g}→{sbp[-1]:g}mmHg" if sbp[0] - sbp[-1] >= 5 else f"수축기 혈압 {sbp[-1]:g}mmHg")
        if mp and mp[-1] <= 70: vit_long.append(f"평균 동맥압 {mp[0]:g}→{mp[-1]:g}mmHg" if mp[0] - mp[-1] >= 5 else f"평균 동맥압 {mp[-1]:g}mmHg")
    low_o2 = bool(spo2 and spo2[-1] <= 92)
    if low_o2:
        vit_short.append("산소포화도 저하"); vit_long.append(f"산소포화도 {spo2[0]:g}→{spo2[-1]:g}%" if spo2[0] - spo2[-1] >= 2 else f"산소포화도 {spo2[-1]:g}%")
    if tmp and (tmp[-1] >= 38.3 or tmp[-1] < 36.0):
        vit_short.append("발열" if tmp[-1] >= 38.3 else "저체온"); vit_long.append(f"체온 {tmp[-1]:g}℃")

    # ── 검사 요약 (장기 기능 / 염증) ──
    def lab_txt(a):
        arrow = "↑" if a["flag"] == "HIGH" else "↓"
        trend = {"rising": ", 상승 중", "falling": ", 하강 중"}.get(a["trend"] or "", "")
        unit = a["unit"] or ""
        sep = "" if unit in ("%", "") else " "
        return f"{LAB_KO.get(a['name'], a['name'])} {a['value']:g}{sep}{unit}{arrow}" + (f"({trend[2:]})" if trend else "")
    organ_names = ["Lactate", "Platelet", "Creatinine", "Bilirubin", "BUN", "Bicarbonate"]
    infl_names = ["WBC", "Neutrophils", "Bands", "CRP"]
    organ = [ab[n] for n in organ_names if n in ab]
    infl = [ab[n] for n in infl_names if n in ab]
    ORGAN_SHORT = {"Lactate": lambda a: f"젖산 {a['value']:g} 상승", "Platelet": lambda a: "혈소판 감소",
                   "Creatinine": lambda a: "크레아티닌 상승", "Bilirubin": lambda a: "빌리루빈 상승"}
    organ_short = [ORGAN_SHORT[a["name"]](a) for a in organ if a["name"] in ORGAN_SHORT]

    # ── 감염 근거: 영상 키워드 + 배양 ──
    imgs = sorted([i for i in patient.get("imaging", []) if i["t_report"] <= T and T - i["t_exam"] <= D["win_radio"]],
                  key=lambda i: i["t_report"])
    img_now = imgs[-1] if imgs else None
    img_now_kw = radiology_keywords(img_now["impression"]) if img_now else None
    img_prev = imgs[-2] if len(imgs) >= 2 else None
    img_prev_kw = radiology_keywords(img_prev["impression"]) if img_prev else None
    pend, done, neg = _culture_state(patient, T)
    infect_short = []
    if img_now_kw and img_now_kw["label"]:
        infect_short.append(f"영상 판독문의 {img_now_kw['label']}")
    infect_short += [d.split("(")[0] for d in done]

    # ── 경과 (과거 점수만) ──
    hist = sorted([h for h in (history or []) if h[0] <= T])
    course, alert_hours, rise_cause = [], None, ""
    if len(hist) >= 2:
        runs = []                          # [tier, start, end, min, max]
        for t, s, tr in hist:
            if runs and runs[-1][0] == tr:
                runs[-1][2] = t; runs[-1][3] = min(runs[-1][3], s); runs[-1][4] = max(runs[-1][4], s)
            else:
                runs.append([tr, t, t, s, s])
        def rng(a, b): return f"{a}" if a == b else f"{a}~{b}"
        def seg(x): return f"{x[1]}~{x[2]}h {TIER_KO[x[0]]}({rng(x[3], x[4])})" if x[1] != x[2] else f"{x[1]}h {TIER_KO[x[0]]}({x[3]})"
        recent = [h for h in hist if h[0] >= T - 12]
        switches = sum(1 for a, b in zip(recent, recent[1:]) if a[2] != b[2])
        if len(runs) == 1:
            course.append(f"{runs[0][1]}~{T}h 계속 {TIER_KO[runs[0][0]]}({rng(runs[0][3], runs[0][4])})입니다.")
        elif switches >= 3 and T - runs[-1][1] >= 2:
            lo, hi = min(h[1] for h in recent), max(h[1] for h in recent)
            course.append(f"최근 12시간 동안 등급이 {switches}번 바뀌며 {lo}~{hi} 사이를 오갑니다. 뚜렷하게 오르거나 내리지 않습니다.")
        else:
            last = runs[-1]
            prev = runs[-4:-1]
            course.append(" → ".join(seg(x) for x in prev) + f" → **{last[1]}h {TIER_KO[last[0]]}({hist[[h[0] for h in hist].index(last[1])][1]}) 진입**")
            if last[2] > last[1]:
                course.append(f"이후 {T - last[1]}시간째 {TIER_KO[last[0]]}({rng(last[3], last[4])}) 상태가 이어지고 있습니다.")
            # 등급이 바뀐 시점 직전 1시간 안에 새로 들어온 입력 → 원인 후보
            tc = last[1]
            arr_names = {l["name"] for l in labs_in if tc - 1 < l["t_result"] <= tc}
            arrived = [LAB_KO.get(n, n) for n in sorted(arr_names, key=lambda n: LAB_PRIORITY.index(n) if n in LAB_PRIORITY else 99)]
            img_arr = [i for i in patient.get("imaging", []) if tc - 1 < i["t_report"] <= tc]
            cul_arr = [c for c in patient.get("cultures", []) if c.get("t_resulted") is not None and tc - 1 < c["t_resulted"] <= tc]
            items = []
            if arrived: items.append(", ".join(arrived[:3]) + (" 등" if len(arrived) > 3 else "") + " 검사 결과")
            if img_arr: items.append("영상 판독")
            if cul_arr: items.append("배양 결과")
            if items: rise_cause = f"{tc}h 직전에 {' · '.join(items)}가 들어왔습니다."
        if rise_cause: course.append(rise_cause)
        past6 = [h for h in hist if h[0] == T - 6]
        if past6 and len(runs) > 1 or (past6 and len(runs) == 1):
            d6 = hist[-1][1] - past6[0][1]
            course.append(f"최근 6시간 점수: {past6[0][1]} → {hist[-1][1]}, " +
                          ("오르는 중" if d6 >= 10 else "내려가는 중" if d6 <= -10 else "비슷한 수준"))
        if tier == "HIGH":
            alert_hours = T - runs[-1][1]
    peak = max(hist, key=lambda h: h[1]) if hist else None

    # ── ① 요약 ──
    if not has_labs:
        s0 = (f"검사 기록이 없어 바이탈만으로 판단했습니다. " +
              (", ".join(vit_short) + " 소견으로 " if vit_short else "") + f"{ft}입니다.")
    elif r["conflict"] and vt == "HIGH":
        s0 = (f"바이탈({', '.join(vit_long[:2]) or '악화'})은 나빠졌지만 검사·영상에 감염 근거가 약해 "
              + ("고위험을 유지했습니다." if tier == "HIGH" else f"{ft}{_ro(ft)} 조정되었습니다.") + " 바이탈 변화의 다른 원인 감별이 필요합니다.")
    elif r["conflict"] and et == "HIGH":
        s0 = (f"바이탈은 아직 안정적이지만 {', '.join(organ_short) or '검사 이상'}으로 검사 악화가 뚜렷해 {ft}{_ro(ft)} 판정했습니다."
              if risk(tier) else "검사 이상이 있지만 바이탈이 안정적이어서 저위험으로 판정했습니다.")
    elif risk(tier):
        parts = infect_short + organ_short[:3] + (["·".join(vit_short[:3]) + " 진행"] if vit_short else [])
        s0 = (", ".join(parts) + ". 이 근거들이 함께 나타나 " if parts else "") + \
             f"4시간 안 패혈증 위험이 {'높습니다' if tier == 'HIGH' else '중간 수준입니다'}."
    elif peak and peak[2] == "HIGH":
        down = next((h[0] for h in hist if h[0] > peak[0] and h[2] != "HIGH"), None)
        s0 = f"앞서 고위험(최고 {peak[1]})이었다가 {down}h 이후 내려와 현재 저위험({round(r['fusion_score'] * 100)})입니다."
        if img_now_kw and img_now_kw["improved"]: s0 += " 영상 판독문의 호전 소견과 맞는 흐름입니다."
    else:
        s0 = "바이탈과 검사에 뚜렷한 위험 신호가 없어 저위험입니다."
        if r["guideline"]["rule_signal"] == "강함": s0 += " 다만 Sepsis-3 관련 지표 일부는 확인이 필요합니다."

    # ── ② 주요 근거 ──
    ev = []
    inf = []
    if img_now:
        lab_ = img_now_kw["label"] or ("이전 음영 호전 소견" if img_now_kw["improved"] else
                                       "폐렴 소견 없음" if any(RAD_KW_SPECIFIC.get(k) == "폐렴" for k in img_now_kw["negated"]) else
                                       "급성 소견 없음" if "no acute" in img_now["impression"].lower() else "")
        if img_now_kw["nonspecific"] and not img_now_kw["label"]:
            lab_ = (lab_ + ", " if lab_ else "") + "비특이 소견 " + "·".join(RAD_KW_NONSPECIFIC[k] for k in img_now_kw["nonspecific"])
        kw = f"({', '.join(img_now_kw['keys'])})" if img_now_kw["keys"] else ""
        inf.append(f"영상{(': ' + lab_ + kw) if lab_ else ''} — \"{img_now['impression']}\" ({round(T - img_now['t_report'])}시간 전 판독)")
        if img_prev and img_prev_kw:
            pl = img_prev_kw["label"] or ("급성 소견 없음" if "no acute" in img_prev["impression"].lower() else "")
            if pl: inf.append(f"이전 영상({round(img_prev['t_exam'])}h 촬영): {pl}")
    inf += [d for d in done]
    inf += [f"{p.split('(')[0]} {p.split('(')[1].split('시간')[0]}시간째 결과 대기" for p in pend]
    inf += [f"{n} 균 자람 없음" for n in neg]
    ev.append("- **감염**: " + (" · ".join(inf) if inf else "배양·영상에 뚜렷한 감염원 근거 없음"))
    if has_labs:
        if organ: ev.append("- **장기 기능**: " + " · ".join(lab_txt(a) for a in organ))
        if infl: ev.append("- **염증**: " + " · ".join(lab_txt(a) for a in infl))
        if not organ and not infl: ev.append("- **검사**: 최근 48시간 정상 범위")
    else:
        ev.append("- **검사**: 기록 없음")
    ev.append("- **바이탈**: " + (", ".join(vit_long) if vit_long else "뚜렷한 이상 없음"))

    # ── ③ 에이전트 판단 ──
    if r["conflict"] and vt == "HIGH":
        s_agent = f"바이탈은 고위험, EMR은 저위험으로 **충돌**했습니다. 합동 심문이 검사 소견이 약하다는 점을 반영해 최종 {ft}{_ro(ft)} 판정했습니다."
    elif r["conflict"] and et == "HIGH":
        s_agent = f"EMR은 고위험, 바이탈은 저위험으로 **충돌**했습니다. 합동 심문이 검사 악화를 반영해 최종 {ft}{_ro(ft)} 판정했습니다."
    elif not has_labs:
        s_agent = f"EMR 근거가 없어 합동 심문이 바이탈에 더 의존했습니다 (바이탈 {TIER_KO[vt]})."
    elif vt == et:
        s_agent = f"바이탈과 EMR이 모두 {TIER_KO[vt]}으로 같은 방향입니다." if vt != "MODERATE" else "바이탈과 EMR이 모두 중간위험으로 같은 방향입니다."
    elif risk(vt) == risk(et):
        s_agent = f"바이탈({TIER_KO[vt]})과 EMR({TIER_KO[et]})이 같은 방향입니다."
    else:
        s_agent = f"바이탈은 {TIER_KO[vt]}, EMR은 {TIER_KO[et]}이며, 합동 심문이 두 근거를 함께 따져 최종 {ft}{_ro(ft)} 판정했습니다."

    # ── ④ Sepsis-3 대조 ──
    g = r["guideline"]
    QKO = {"RR": "호흡수", "SBP": "수축기 혈압"}
    q_txt = ", ".join(f"{QKO.get(x.split()[0], x.split()[0])} {x.split()[1]}" for x in g["qsofa_items"])
    FL = {"Lactate": "젖산 2 초과", "Platelet": "혈소판 감소", "Creatinine": "크레아티닌 상승", "Bilirubin": "빌리루빈 상승", "WBC": "백혈구 이상"}
    fl = [FL.get(f["name"], f["name"]) for f in g["lab_flags"]]
    met = ([f"qSOFA {g['qsofa_count']}개 충족({q_txt})"] if g["qsofa_count"] else []) + fl
    if not met:
        s3 = "qSOFA·장기 기능 지표 모두 충족 없음" + (" → 모델 단독 신호입니다." if risk(tier) else ".")
    else:
        head = (", ".join(met[:-1]) + f"에 {met[-1]}" if len(met) > 1 else met[0])
        if not g["lab_flags"] and g["qsofa_count"] and has_labs:
            tail = "이나 장기 기능 검사 이상 없음 → 바이탈만의 신호입니다."
        elif risk(tier):
            tail = {"강함": "이 겹쳐 모델 판단과 일치합니다.", "약함": " — 약한 신호라 모델 판단을 뒷받침하는 정도는 제한적입니다.",
                    "없음": ""}[g["rule_signal"]]
        else:
            tail = " → 모델은 저위험이지만 재확인이 필요합니다." if g["rule_signal"] == "강함" else " — 약한 신호입니다."
        s3 = head + tail
    s3 += " (의식 상태는 qSOFA 계산에서 제외)" if g["qsofa_count"] else ""

    # ── ⑤ 확인 권고: 근거 → 항목 ──
    todo = []
    if tier == "HIGH":
        todo.append(f"고위험 경보{f' {alert_hours}시간째' if alert_hours else ''} → 책임의사 환자 확인 (3시간 안)")
    elif tier == "MODERATE":
        todo.append("중간위험 → 관찰 목록 등록 후 다음 측정에서 재평가")
    else:
        todo.append("저위험 → 일상 모니터링 유지")
    if risk(tier):
        if "Lactate" in ab and ab["Lactate"]["flag"] == "HIGH":
            todo.append(f"젖산 {ab['Lactate']['value']:g}{' 상승 중' if ab['Lactate']['trend'] == 'rising' else ''} → 젖산 재측정 고려")
        if bp_low:
            todo.append(f"수축기 혈압 {sbp[-1]:g}" + (f", 평균 동맥압 {mp[-1]:g}" if mp and mp[-1] <= 70 else "") + "으로 하강 → 혈압 추이 확인")
        if low_o2:
            todo.append(f"산소포화도 {spo2[-1]:g}%" + (", 폐렴 시사 영상" if img_now_kw and "폐렴" in (img_now_kw["label"] or "") else "") + " → 산소화 상태 확인")
        if done: todo.append(f"{done[0].split('에서 ')[1].split(' 검출')[0]} 검출 → 감수성 결과 확인")
        for p in pend:
            todo.append(f"{p.split('(')[0]} 결과 대기 → 결과 확인")
        if not (pend or done or neg): todo.append("배양 기록 없음 → 감염이 의심되면 배양 채취 여부 검토")
        if not has_labs: todo.append("검사 기록 없음 → 기본 혈액검사(젖산·혈구·신기능) 시행 여부 검토")
        if r["conflict"] and vt == "HIGH": todo.append("바이탈만 나쁨 → 통증·출혈·수술 후 반응 등 다른 원인 감별")
        if r["conflict"] and et == "HIGH": todo.append("바이탈은 안정적 → 측정 간격 단축 고려")
    else:
        if "CRP" in ab: todo.append("CRP 아직 높음 → 염증 지표 추세 확인")
        if g["rule_signal"] == "강함": todo.append("Sepsis-3 관련 지표 위험 신호 → 재확인")

    # ── ⑥ 판단 한계 (해당할 때만) ──
    limits = []
    if not has_labs: limits.append("검사 기록이 없어 EMR 근거 없이 바이탈 위주로 판단했습니다.")
    n_slot = len(r["vital_window"]); n_meas = len(hr)
    if n_slot and n_meas < n_slot * 0.6:
        limits.append(f"바이탈 측정 간격이 길어(직전 {n_slot}시간 중 {n_meas}회 측정) 변화 반영이 제한적입니다.")

    L = ["### 요약", f"**{s0}**", ""]
    if course: L += ["### 경과"] + [f"- {c}" for c in course] + [""]
    L += ["### 주요 근거"] + ev + [""]
    L += ["### 에이전트 판단", s_agent, ""]
    L += ["### Sepsis-3 대조", s3, ""]
    L += ["### 확인 권고 (참고용)"] + [f"- {t}" for t in todo] + [""]
    if limits: L += ["### 판단 한계"] + [f"- {x}" for x in limits] + [""]
    L += ["> 요약·근거 문장은 규칙 기반으로 생성되며, 영상 소견은 판독문 키워드 추출 결과입니다. 확인 권고는 참고용이며 진단·처방을 대체하지 않습니다.", ""]
    return [x.replace("~", "–") for x in L]   # 마크다운에서 ~ 두 개가 취소선으로 렌더링되는 것 방지


def report_md(r: dict, patient: dict, history: list | None = None) -> str:
    th = CFG["thresholds"]["fusion"]
    L = [f"## {TIER_KO[r['tier']]} · 위험 점수 {round(r['fusion_score'] * 100)}",
         f"- 기준 시점: 입실 후 {r['T']}시간 · 4시간 이내 패혈증 발병 위험", ""]
    L += narrative_md(r, patient, history)
    L += [NARR_MARK, "",
         f"- 등급 기준: 보정 전 상대 점수, 고위험 ≥ {round(th['high'] * 100)} · 중간위험 ≥ {round(th['moderate'] * 100)} · 등급 권고: {ACTION[r['tier']]}", "",
         "### 에이전트 판단",
         f"- 바이탈: {TIER_KO[r['vital_tier']]} (점수 {r['vital_score']:.3f}) · EMR: {TIER_KO[r['emr_tier']]} (점수 {r['emr_score']:.3f})"
         + (" · **두 에이전트 판단 충돌**" if r["conflict"] else ""),
         f"- 에이전트 의존도 (attention): 바이탈 {r['agent_dependency']['vital'] * 100:.0f}% · EMR {r['agent_dependency']['emr'] * 100:.0f}%", ""]
    # 바이탈 근거: 창 처음 대비 마지막 변화 + attention 상위 시간대
    win = [w for w in r["vital_window"] if w]
    if win:
        L.append("### 바이탈 근거")
        first, lastw = win[0], win[-1]
        for k in ["HR", "RR", "SBP", "SpO2", "Temp"]:
            a, b = first.get(k), lastw.get(k)
            if a is not None and b is not None:
                L.append(f"- {VNAME[k]}: {a:g} → {b:g} (창 {first['h']}h → {lastw['h']}h)")
        att = np.array(r["vital_attention"]); top = [int(i) for i in np.argsort(-att)[:3] if r["vital_window"][int(i)]]
        if top:
            L.append("- 결정적 시간대 (attention 상위): " + ", ".join(f"{r['vital_window'][i]['h']}h" for i in sorted(top)))
        L.append("")
    L.append("### EMR 근거")
    if r["abnormal_labs"]:
        for a in r["abnormal_labs"]:
            L.append(f"- {a['name']} {a['value']:g} {a['unit'] or ''} ({a['flag']}{', ' + a['trend'] if a['trend'] else ''}, {a['hours_ago']:g}h 전)")
    else:
        L.append("- 정상 범위를 벗어난 검사 없음 (최근 48h)")
    for c in r["cultures"]:
        res = "결과 대기" if c["status"] == "pending" else (c["organisms"] or "균 보고 없음")
        L.append(f"- 배양 {c['spec_type_desc']} ({c['hours_ago']:g}h 전 채취): {res}")
    if r["imaging"]:
        L.append(f"- 영상 판독 있음 ({r['imaging']['hours_ago']:g}h 전 촬영)")
    if r["emr_section_effect"]:
        eff = {k: v for k, v in r["emr_section_effect"].items() if v is not None}
        if eff:
            L.append("- EMR 섹션 기여도: " + ", ".join(f"{ {'labs': '검사', 'cultures': '배양', 'imaging': '영상'}[k] } {-v:+.3f}" for k, v in eff.items()))
    L.append("")
    g = r["guideline"]
    L.append("### Sepsis-3 관련 지표 (규칙 플래그)")
    L.append("- 검사: " + (", ".join(f"{f['name']} {f['value']:g} ({f['meaning']})" for f in g["lab_flags"]) or "해당 없음"))
    L.append("- qSOFA 항목: " + (", ".join(g["qsofa_items"]) or "해당 없음") + f" ({g['qsofa_count']}/2, 의식 상태 미포함)")
    L.append(f"- 규칙 신호: {g['rule_signal']}")
    L.append("")
    L.append("> 연구용 시연 · 가상 환자 · 진단을 대체하지 않음")
    return "\n".join(L)
