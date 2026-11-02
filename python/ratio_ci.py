#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ratio_ci.py -- 为全量图中的比例型结果补置信区间

为什么不能直接用二项区间
------------------------
链路在楼层内相关（同一楼层的几何相似），独立同分布假设不成立，朴素的二项区间
会低估不确定性。本脚本同时给出三组数：

1. pool 比例与独立假设下的 Wilson 95% 区间（下界参考，偏窄）
2. 按 split 的 11 个比例值及其最小、中位、最大（跨数据集变异性）
3. 以 split 为簇的聚类自助 95% 区间（重采样 split，按样本量加权）

用法
----
python ratio_ci.py
输出 results/ratio_ci.csv 与终端表格。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
COLS = ["rank", "kappa", "d_eff", "share_z"]

METRICS = {
    "rank3": (lambda d: d["rank"] == 3),
    "kappa_lt100": (lambda d: d["kappa"] < 100),
    "deff_lt2": (lambda d: d["d_eff"] < 2),
    "sharez_gt05": (lambda d: d["share_z"] > 0.5),
    "sharez_gt13": (lambda d: d["share_z"] > 1.0 / 3.0),
    "strict_false_pass": (lambda d: (d["rank"] == 3) & (d["kappa"] < 100)
                          & (d["share_z"] > 0.5)),
}


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / den, (c + h) / den)


def main() -> int:
    files = sorted(RES.glob("c2full_*.csv"))
    if not files:
        print("未找到 results/c2full_*.csv")
        return 1
    split_names, cnt, tot = [], {m: [] for m in METRICS}, []
    for p in files:
        d = pd.read_csv(p, usecols=COLS, dtype="float32")
        n = len(d)
        split_names.append(p.stem[len("c2full_"):])
        tot.append(n)
        for m, fn in METRICS.items():
            cnt[m].append(int(fn(d).sum()))
        del d
    tot = np.asarray(tot, dtype=float)
    K = len(files)

    rng = np.random.default_rng(20261012)
    B = 4000
    rows = []
    for m in METRICS:
        c = np.asarray(cnt[m], dtype=float)
        pooled = c.sum() / tot.sum()
        lo, hi = wilson(c.sum(), tot.sum())
        per = c / tot
        boot = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, K, K)
            boot[b] = c[idx].sum() / tot[idx].sum()
        rows.append(dict(
            metric=m, n=int(tot.sum()), pool=pooled,
            wilson_lo=lo, wilson_hi=hi,
            split_min=float(per.min()), split_med=float(np.median(per)),
            split_max=float(per.max()),
            cluster_lo=float(np.percentile(boot, 2.5)),
            cluster_hi=float(np.percentile(boot, 97.5)),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "ratio_ci.csv", index=False, float_format="%.6f")

    print(f"总链路 {int(tot.sum()):,}，划分 {K} 个")
    with pd.option_context("display.width", 160, "display.float_format",
                           "{:.4f}".format):
        print(out.to_string(index=False))
    print()
    print("按 split 的比例明细：")
    for m in METRICS:
        c = np.asarray(cnt[m], dtype=float)
        line = ", ".join("%s %.4f" % (s.split("_", 1)[-1], v)
                         for s, v in zip(split_names, c / tot))
        print(f"  {m:20s} {line}")
    print(f"\n写出 {RES / 'ratio_ci.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
