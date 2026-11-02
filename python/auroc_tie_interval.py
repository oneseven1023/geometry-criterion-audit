# -*- coding: utf-8 -*-
"""领先判据量的 AUROC 并列性检验（回应「预测优势是否全面支持」）。

表 IV 的前两名是等效虚拟锚点方向的垂直分量 mean|u_z| 与条件数 kappa，取值
0.706 与 0.703。单看这两个数无法判断差异是否有意义，本脚本按论文自身的统计
单位（以划分为簇）做自助法，给出各自的区间与配对差值区间。

目标变量 y = 1[AOA 估计器垂直误差 aoa_tz > 0.5 m]，样本为端到端对照的
33,086 个接收机位置（前 30,086 为 WiPTR，另 3,000 取自两个 Wi3R 划分）。

结果：mean|u_z| 0.706 [0.669, 0.792]，kappa 0.703 [0.630, 0.765]，
配对差值 [-0.056, +0.068]，为正的重复占 0.60。即两者统计上并列。
注意簇数只有 5（端到端样本的划分个数），区间偏宽，这正是需要如实报告的。

用法：项目根目录下运行
    python python/auroc_tie_interval.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
LINKS = ROOT / "results" / "e2e_links.csv"

B = 4000
SEED = 7


def main() -> int:
    d = pd.read_csv(LINKS)
    y = (d["aoa_tz"].to_numpy(dtype=float) > 0.5).astype(int)

    # 统一取向为「越大越是垂直方向更坏」
    uz = -d["mean_abs_uz"].to_numpy(dtype=float)
    ka = d["kappa"].to_numpy(dtype=float)
    sz = d["share_z"].to_numpy(dtype=float)

    print(f"n={len(d)}  pos_rate={y.mean():.4f}")
    for name, v in (("mean|u_z|(-1)", uz), ("kappa(+1)", ka), ("s_z(+1)", sz)):
        print(f"  oriented AUROC {name:14s} {roc_auc_score(y, v):.4f}")

    splits = d["split"].to_numpy()
    uniq = np.unique(splits)
    idx = {s: np.where(splits == s)[0] for s in uniq}
    frame = pd.DataFrame({"y": y, "uz": uz, "ka": ka})

    rng = np.random.default_rng(SEED)
    au, ak, diffs = [], [], []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        sub = frame.iloc[rows]
        if sub["y"].nunique() < 2:
            continue
        a = roc_auc_score(sub["y"], sub["uz"])
        k = roc_auc_score(sub["y"], sub["ka"])
        au.append(a)
        ak.append(k)
        diffs.append(a - k)

    au, ak, diffs = np.array(au), np.array(ak), np.array(diffs)
    print(f"\n簇数={len(uniq)}  有效重复={len(diffs)}")
    print(f"mean|u_z| 95% CI [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]")
    print(f"kappa    95% CI [{np.percentile(ak, 2.5):.3f}, {np.percentile(ak, 97.5):.3f}]")
    print(f"配对差值 95% CI [{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}]"
          f"  中位 {float(np.median(diffs)):.4f}  P(>0)={float(np.mean(diffs > 0)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
