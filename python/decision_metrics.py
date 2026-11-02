#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""decision_metrics.py -- 方向诊断指标的公平基线与决策价值评估

要回答的问题
------------
审稿意见 P0-1：新增的 s_z、d_eff 相对既有标量（GDOP、D-optimality、最小特征值、
完整协方差对角）究竟增加了什么？
审稿意见 P1-2：把部署规则当作决策任务，指标能否预测「垂直误差是否超限」？

方法
----
对 `results/e2e_links.csv` 的每个接收机位置，取实际垂直误差作为结果变量
y = 1[垂直总误差 > tau]。对每个候选标量 x（统一取向为「越大越坏」），
计算
  AUROC     排序能力，0.5 表示无区分度
  AP        PR 曲线下面积（不平衡更敏感）
  TPR/FPR   在训练划分上按 Youden 指数定阈，迁移到留出划分后的表现
并给出 Spearman 相关与分位对照。

候选标量
--------
传统判据侧  kappa（条件数）、logdet（D-optimality 代理）、gdop、lam_min、lam_max
方向判据侧  d_eff（取负，越大越坏）、share_z、mean|u_z|（取负）、share_x、share_y

输出
----
results/decision_metrics.csv     逐指标的 AUROC/AP/迁移表现
results/decision_sweep.csv       share_z 阈值扫描（决策曲线）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RES = Path(__file__).resolve().parent.parent / "results"

# 预测器：名称 -> (列名, 符号)。符号 +1 表示列值越大越坏，-1 表示越小越坏。
PREDICTORS = [
    ("kappa", "kappa", +1),
    ("logdet", "logdet", -1),
    ("gdop", "gdop", +1),
    ("lam_min", "lam_min", -1),
    ("lam_max", "lam_max", -1),
    ("d_eff", "d_eff", -1),
    ("share_z", "share_z", +1),
    ("share_y", "share_y", -1),
    ("share_x", "share_x", -1),
    ("mean|u_z|", "mean_abs_uz", -1),
    ("VDOP", "vdop", +1),
]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def youden_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """在训练集上按 Youden 指数 (TPR-FPR) 取阈值。"""
    order = np.argsort(score)
    s = score[order]
    yy = y[order]
    npos = yy.sum()
    nneg = len(yy) - npos
    if npos == 0 or nneg == 0:
        return float(np.median(score))
    # 累计正例比例与负例比例随阈值从低到高前进
    cum_pos = np.cumsum(yy) / npos
    cum_neg = np.cumsum(1 - yy) / nneg
    j = cum_pos - cum_neg
    k = int(np.argmax(j))
    return float(s[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path, default=RES / "e2e_links.csv")
    ap.add_argument("--tau-z", type=float, default=0.5,
                    help="垂直误差容忍限 (m)")
    ap.add_argument("--out", type=Path, default=RES / "decision_metrics.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.links)
    splits = sorted(df["split"].unique())
    rows = []

    for est, ecol in (("aoa", "aoa_tz"), ("wls", "wls_tz")):
        d = df[df[ecol].notna()].copy()
        # 垂直精度因子：由信息矩阵隐含的垂直坐标标准差
        d["vdop"] = np.sqrt(d["share_z"].to_numpy(dtype=float)
                            * d["trace_inv"].to_numpy(dtype=float))
        y = (d[ecol].to_numpy() > args.tau_z).astype(int)
        for name, col, sign in PREDICTORS:
            x = sign * d[col].to_numpy(dtype=float)
            m = np.isfinite(x)
            if m.sum() < 50 or y[m].sum() == 0:
                continue
            xm, ym = x[m], y[m]
            n = m.sum()
            # 池化
            try:
                auroc = float(roc_auc_score(ym, xm))
                aprec = float(average_precision_score(ym, xm))
            except Exception:
                continue
            rho = spearman(xm, d[ecol].to_numpy()[m])
            # 逐划分 AUROC，再按链路数加权
            per, w = [], []
            for sp in splits:
                mm = (d["split"].to_numpy() == sp) & m
                if mm.sum() < 30:
                    continue
                ys = y[mm]
                if ys.sum() == 0 or ys.sum() == ys.size:
                    continue
                per.append(roc_auc_score(ys, x[mm]))
                w.append(int(mm.sum()))
            auc_split = float(np.average(per, weights=w)) if per else np.nan
            # 留一划分：训练集定阈，留出集评估
            tprs, fprs = [], []
            for sp in splits:
                te = (d["split"].to_numpy() == sp) & m
                tr = (~te) & m
                if te.sum() < 30 or tr.sum() < 30:
                    continue
                if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                    continue
                thr = youden_threshold(x[tr], y[tr])
                pred = x[te] >= thr
                yt = y[te]
                tp = np.sum(pred & (yt == 1))
                fp = np.sum(pred & (yt == 0))
                tprs.append(tp / max(np.sum(yt == 1), 1))
                fprs.append(fp / max(np.sum(yt == 0), 1))
            rows.append(dict(
                estimator=est, metric=name, n=int(n), pos_rate=float(ym.mean()),
                auroc=auroc, avg_prec=aprec, spearman=rho,
                auroc_split_wtd=auc_split,
                transfer_tpr=float(np.mean(tprs)) if tprs else np.nan,
                transfer_fpr=float(np.mean(fprs)) if fprs else np.nan,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    for est in ("aoa", "wls"):
        sub = out[out.estimator == est].sort_values("auroc", ascending=False)
        print(f"\n== 估计器 {est.upper()}  (tau_z = {args.tau_z} m) ==")
        print(sub.drop(columns=["estimator"]).to_string(index=False))

    # ---- share_z 阈值扫描（决策曲线） ----
    sweep = []
    for est, ecol in (("aoa", "aoa_tz"), ("wls", "wls_tz")):
        d = df[df[ecol].notna()]
        y = (d[ecol].to_numpy() > args.tau_z).astype(int)
        for thr in np.arange(0.20, 0.81, 0.05):
            pred = (d["share_z"].to_numpy() >= thr)
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            fn = np.sum(~pred & (y == 1))
            tn = np.sum(~pred & (y == 0))
            sweep.append(dict(estimator=est, threshold=round(float(thr), 2),
                              tpr=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                              precision=tp / max(tp + fp, 1),
                              flagged=float(pred.mean())))
    sw = pd.DataFrame(sweep)
    sw.to_csv(RES / "decision_sweep.csv", index=False, float_format="%.4f")
    print(f"\n写出 {args.out} 与 {RES / 'decision_sweep.csv'}")
    print("\n== share_z 阈值扫描（AOA） ==")
    print(sw[sw.estimator == "aoa"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
