"""
fusion_06a_swap_emr.py — 융합 데이터의 EMR 벡터만 다른 EMR 모델 것으로 교체 (바이탈 벡터·그리드·라벨 그대로)
  입력 : --emr  train_emr.py --export_from 결과의 접두어 (예: preds/qwen3_1p7b_full → preds/qwen3_1p7b_full_vec_{train,val,test}.npz)
         --src  기존 융합 데이터 (기본 data/fusion_data_v3 = 바이탈 v3)
  출력 : --out/{train,val,test}.npz (키는 fusion_data_v3 와 동일, emr_vec 차원만 모델에 따라 다름)
  L48 문서와 k3 문서의 (stay_id, t_point, split, label) 그리드는 452,214 행 완전 일치 (확인함) → 미매칭 0 이어야 정상
실행 : python fusion_06a_swap_emr.py --emr <접두어> --out data/fusion_data_<이름>
       python fusion_06b_v4.py --data data/fusion_data_<이름> --name <이름> --seed 42   (seed 1, 2 도)
"""
import os, sys, json, argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--emr", required=True)
ap.add_argument("--src", default=r"./data/fusion_data_v3")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rep = {"emr": a.emr, "src": a.src}
for s in ["train", "val", "test"]:
    z = dict(np.load(os.path.join(a.src, f"{s}.npz"), allow_pickle=True))
    e = np.load(f"{a.emr}_vec_{s}.npz")
    key = pd.Series(np.arange(len(e["prob"])),
                    index=pd.MultiIndex.from_arrays([e["stay_id"].astype(np.int64), e["t_point"].astype(str)]))
    idx = key.reindex(pd.MultiIndex.from_arrays([z["stay_id"].astype(np.int64), z["t_point"].astype(str)])).to_numpy()
    miss = int(np.isnan(idx).sum()); assert miss == 0, f"{s}: EMR 벡터에 없는 융합 시점 {miss}개"
    idx = idx.astype(np.int64)
    assert (e["label"][idx] == z["label"]).all(), f"{s}: 라벨 불일치"
    z["emr_vec"] = e["vec"][idx].astype(np.float32); z["emr_prob"] = e["prob"][idx].astype(np.float32)
    np.savez_compressed(os.path.join(a.out, f"{s}.npz"), **z)
    rep[s] = {"N": int(len(idx)), "emr_dim": int(z["emr_vec"].shape[1]),
              "emr_auroc": float(roc_auc_score(z["label"], z["emr_prob"])),
              "vital_auroc": float(roc_auc_score(z["label"], z["vital_prob"]))}
    print(f"[{s}] N={len(idx):,} emr_dim={rep[s]['emr_dim']} | 단독 AUROC vital={rep[s]['vital_auroc']:.4f} emr={rep[s]['emr_auroc']:.4f}", flush=True)
json.dump(rep, open(os.path.join(a.out, "build_report.json"), "w"), indent=2, ensure_ascii=False)
print(f"=== 완료: {a.out} ===")
