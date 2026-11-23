r"""
vital_v3_model.py — 융합 빌드용 VitalAgent (vital_v3.py 의 모델 정의를 그대로 복사, import 시 학습 안 돌아감)

최종 run: T4_relax_w24_all (seed 42)
  ckpt     : ./models/v2/vital_T4_relax_w24_all.pt
  FEATURES : 아래 FEATURES (순서 고정 = vital_norm_stats_b0_4_relax.json 키 순서)
  WINDOW   : 24
forward    : model(vital_x, pad_mask, demo_x, delta_t)
  vital_x  (B, W, V)  정규화된 값, 결측 = -1, 앞쪽 패딩 행도 -1
  pad_mask (B, W) bool  True = 패딩 행 (실측 행이 없는 시점) → attention softmax 에서 제외
  demo_x   (B, W, 2)  [age_z, gender_bin] 시점마다 반복
  delta_t  (B, W)     [W, W-1, ..., 1]
주의: pad_mask 는 "행이 없음" 기준. 값이 -1 인지로 만들지 말 것 (실측 행의 결측 피처도 -1).
      pad_mask 가 전부 True 인 윈도우(시점 미매칭)는 softmax 가 NaN → 빌드에서 마지막 행이 실측인지 보장할 것.
"""
import torch, torch.nn as nn, torch.nn.functional as F

MASK = -1.0
WINDOW = 24
FEATURES = ["HR", "SBP", "RR", "SpO2", "Temp", "DBP", "MAP", "SI"]
DEMO = ["age_z", "gender_bin"]
CKPT = r"./models/v2/vital_T4_relax_w24_all.pt"
NORM_JSON = r"./data/v2/vital_norm_stats_b0_4_relax.json"

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

def load(device="cpu"):
    m = VitalAgent(len(FEATURES))
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    return m.to(device).eval()
