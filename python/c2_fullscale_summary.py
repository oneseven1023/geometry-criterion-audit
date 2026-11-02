#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_fullscale_summary.py -- 汇总全量审计结果，输出跨几何图谱与实用后果。

输入：results/c2full_*.csv（由 c2_fullscale_vec.py 产出）。
输出：跨 split 对照表、全局相关性、以及把诊断指标联系到实际精度后果的分析。
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"


def load(path):
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    return header, {k: data[:, i] for i, k in enumerate(header)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="c2full_*.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(str(RES / args.glob)))
    if not files:
        print(f"未找到 {args.glob}"); return

    print(f"{'split':<30}{'n':>9}{'rank3%':>8}{'d_eff':>7}"
          f"{'shz_med':>8}{'shz>0.5%':>9}{'kappa':>8}{'d_eff<2%':>9}{'假通过%':>8}")
    print("-" * 96)
    allc = {}
    for p in files:
        name = os.path.basename(p)[len("c2full_"):-4]
        _, c = load(p)
        n = c["rank"].shape[0]
        r3 = c["rank"]
        de = c["d_eff"]
        sz = c["share_z"]
        ka = np.clip(c["kappa"], 0, 1e12)
        # 假通过：判据报满秩，但垂直方向方差份额超过 0.5
        fp = (r3 == 3) & (sz > 0.5)
        print(f"{name:<30}{n:>9}{100*(r3==3).mean():>7.1f}%"
              f"{np.median(de):>7.3f}{np.median(sz):>8.4f}"
              f"{100*(sz>0.5).mean():>8.1f}%{np.median(ka):>8.1f}"
              f"{100*(de<2).mean():>8.1f}%{100*fp.mean():>7.1f}%")
        for k, v in c.items():
            allc.setdefault(k, []).append(v)
    cat = {k: np.concatenate(v) for k, v in allc.items()}
    N = cat["rank"].shape[0]

    sz = cat["share_z"]; uz = np.abs(cat["mean_abs_uz"])
    de = cat["d_eff"]; ka = np.clip(cat["kappa"], 0, 1e12)
    r3 = cat["rank"]; los = cat["los_frac"]
    lam_max = cat["lam_max"]; var_z = cat["var_z"]

    print(f"\n全局（样本 {N}）：")
    print(f"  秩分布: " + ", ".join(
        f"r{int(r)}:{int((r3==r).sum())}" for r in np.unique(r3)))
    print(f"  秩3 占比 {100*(r3==3).mean():.2f}%")
    print(f"  kappa 中位 {np.median(ka):.1f}   90分位 {np.percentile(ka,90):.1f}")
    print(f"  d_eff 中位 {np.median(de):.3f}   d_eff<2 占比 {100*(de<2).mean():.1f}%")
    print(f"  share_z 中位 {np.median(sz):.4f}   share_z>0.5 占比 {100*(sz>0.5).mean():.1f}%"
          f"   share_z>1/3 占比 {100*(sz>1/3).mean():.1f}%")
    fp_strict = (r3 == 3) & (ka < 100) & (sz > 0.5)
    fp_mid = (r3 == 3) & (ka < 100)
    print(f"  严格假通过(秩3 且 kappa<100 且 share_z>0.5) 占比 {100*fp_strict.mean():.2f}%"
          f"  （秩3 且 kappa<100 占比 {100*fp_mid.mean():.2f}%）")
    print(f"  corr(share_z, mean|u_z|) = {np.corrcoef(sz, uz)[0,1]:+.3f}")
    print(f"  corr(share_z, d_eff)     = {np.corrcoef(sz, de)[0,1]:+.3f}")

    # 各向异性：最差轴与最好轴的标准差比 = sqrt(kappa)
    aniso = np.sqrt(ka)
    print(f"\n各向异性（最差轴/最好轴 的标准差比 = sqrt(kappa)）：")
    print(f"  中位 {np.median(aniso):.2f}   90分位 {np.percentile(aniso,90):.2f}   "
          f"最大值 {aniso[np.isfinite(aniso)].max():.1f}")
    # 垂直方向相对最好轴的标准差倍数 = sqrt(var_z * lam_max)
    vert = np.sqrt(np.clip(var_z * lam_max, 0, None))
    print(f"  垂直方向相对最好轴的标准差倍数：中位 {np.median(vert):.2f}  "
          f"90分位 {np.percentile(vert,90):.2f}")

    # ---- 实用后果：在判据「通过」的条件下，垂直退化有多严重 ----
    print(f"\n实用后果（在判据给出通过的条件子集上）：")
    for kap_thr in (30.0, 100.0, 300.0):
        m = (r3 == 3) & (ka < kap_thr)
        if m.sum() < 100:
            continue
        print(f"  条件 kappa<{kap_thr:.0f}（n={int(m.sum())}，占全体 {100*m.mean():.1f}%）："
              f" share_z 中位 {np.median(sz[m]):.4f}  "
              f"share_z>0.5 占比 {100*(sz[m]>0.5).mean():.1f}%  "
              f"垂直/最好轴标准差倍数 中位 {np.median(vert[m]):.2f}")

    # 分层：LoS
    print(f"\n按视距分层：")
    for lo, hi, nm in [(0.99, 1.01, "全LoS"), (0.0, 0.01, "全NLoS"), (0.01, 0.99, "混合")]:
        m = (los >= lo) & (los < hi)
        if m.sum() > 100:
            print(f"  [{nm}] n={int(m.sum())}  share_z 中位 {np.median(sz[m]):.4f}"
                  f"  d_eff 中位 {np.median(de[m]):.3f}  kappa 中位 {np.median(ka[m]):.1f}")

    # 分层：锚点数
    print(f"\n按锚点数分层：")
    na = cat["n_anchor"]
    for v in np.unique(na):
        m = na == v
        if m.sum() > 100:
            print(f"  [锚点{int(v)}] n={int(m.sum())}  share_z 中位 {np.median(sz[m]):.4f}"
                  f"  d_eff 中位 {np.median(de[m]):.3f}  秩3占比 {100*(r3[m]==3).mean():.1f}%")

    # 分层：接收机高度
    print(f"\n按接收机高度层分层（前若干高度）：")
    rz = cat["rx_z"]
    for z in np.unique(np.round(rz, 2))[:8]:
        m = np.round(rz, 2) == z
        if m.sum() > 100:
            print(f"  [rx_z={z:.2f}] n={int(m.sum())}  share_z 中位 {np.median(sz[m]):.4f}"
                  f"  mean|u_z| 中位 {np.median(uz[m]):.4f}")


if __name__ == "__main__":
    main()
