#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_summary.py -- 汇总 C2 最小实验明细，输出跨几何的对照表与关键相关性。"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    def col(k):
        return np.array([float(r[k]) for r in rows])
    return rows, col


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="c2_*.csv", help="results 目录下要汇总的明细文件模式")
    args = ap.parse_args()
    files = sorted(glob.glob(str(RES / args.glob)))
    if not files:
        print(f"未找到匹配 {args.glob} 的明细文件"); return
    print(f"{'split':<22}{'n':>6}{'rank3%':>8}{'d_eff':>8}{'shz_med':>9}"
          f"{'shz>0.5%':>10}{'kappa':>9}{'d_eff<2%':>10}")
    print("-" * 82)
    all_rows = []
    for p in files:
        name = os.path.basename(p)[:-4]
        if "_" in name:
            name = name.split("_", 1)[1]
        rows, col = load(p)
        n = len(rows)
        r3 = col("rank_ideal")
        de = col("d_eff_both")
        sz = col("share_z_both")
        ka = col("kappa_ideal")
        print(f"{name:<22}{n:>6}{100*(r3==3).mean():>7.1f}%"
              f"{np.median(de):>8.3f}{np.median(sz):>9.4f}"
              f"{100*(sz>0.5).mean():>9.1f}%{np.median(ka):>9.1f}"
              f"{100*(de<2).mean():>9.1f}%")
        all_rows.extend(rows)

    # 全局相关性：垂直份额 与 到达方向垂直分量
    uz = np.array([float(r["mean_abs_uz"]) for r in all_rows])
    sz = np.array([float(r["share_z_both"]) for r in all_rows])
    los = np.array([float(r["los_frac"]) for r in all_rows])
    de = np.array([float(r["d_eff_both"]) for r in all_rows])
    print("\n全局相关性（Pearson，样本 %d）：" % len(all_rows))
    print(f"  corr(share_z, mean|u_z|)   = {np.corrcoef(sz, uz)[0,1]:+.3f}   "
          f"（越负说明到达方向越水平则垂直越不可辨识）")
    print(f"  corr(share_z, d_eff)       = {np.corrcoef(sz, de)[0,1]:+.3f}")
    print(f"  全LoS  share_z 中位 {np.median(sz[los>0.99]) if (los>0.99).any() else float('nan'):.4f} "
          f"  d_eff 中位 {np.median(de[los>0.99]) if (los>0.99).any() else float('nan'):.3f}")
    m = los < 0.5
    print(f"  多数NLoS share_z 中位 {np.median(sz[m]) if m.any() else float('nan'):.4f} "
          f"  d_eff 中位 {np.median(de[m]) if m.any() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
