#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_bandwidth_scan.py -- 跨带宽的判据失效扫描（论文图 4(b) 的数据源）

背景
----
全量图谱 `c2_fullscale_vec.py` 只跑 B = 100 MHz 的判据侧（理想 FIM 与动态范围
受限侧），分辨率受限口径是逐链路 O(n^2) 合并，不能在 461 万条上跑。因此跨带宽
这一维度没有全量数据，只有试点（val_grid 两层）给出的定性结论。

本脚本在少量代表性 split 上，用带分辨率合并的逐链路脚本 `c2_criterion_audit`
在 20 / 100 / 400 MHz 三个带宽上各跑一遍，输出每 (split, 带宽) 的聚合量，供
论文图 4(b) 使用。口径与试点、与全量图谱完全一致（角分辨率 20 度、动态范围
30 dB、按接收机真实坐标分组）。

用法
----
python c2_bandwidth_scan.py --floors 2 --max-rx 400
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_criterion_audit import run_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"

DEFAULT_SPLITS = ["WiPTR/train", "WiPTR/val_grid", "WiPTR/val_ood",
                  "Wi3R/main_5tx", "Wi3R/outofdistribution/fourrooms"]
DEFAULT_BW = [20e6, 100e6, 400e6]


def agg(rows: list) -> dict:
    col = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    ri, rb = col("rank_ideal"), col("rank_both")
    si, sb = col("share_z_ideal"), col("share_z_both")
    return dict(
        n=len(rows),
        rank3_ideal=float((ri == 3).mean()),
        rank_short_both=float(((ri == 3) & (rb < 3)).mean()),
        share_z_ideal_med=float(np.median(si)),
        share_z_both_med=float(np.median(sb)),
        d_eff_ideal_med=float(np.median(col("d_eff_ideal"))),
        d_eff_both_med=float(np.median(col("d_eff_both"))),
        kappa_med=float(np.median(col("kappa_ideal"))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    ap.add_argument("--bw", default=",".join(str(int(b)) for b in DEFAULT_BW))
    ap.add_argument("--floors", type=int, default=2)
    ap.add_argument("--max-rx", type=int, default=400)
    ap.add_argument("--dtheta", type=float, default=20.0)
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--csv", default=str(RES / "c2bw_scan.csv"))
    args = ap.parse_args()

    want = [s.strip() for s in args.splits.split(",") if s.strip()]
    bws = [float(b) for b in args.bw.split(",")]
    splits = {s.tag: s for s in wio.discover_splits(ROOT)}
    missing = [t for t in want if t not in splits]
    if missing:
        print("未找到 split:", missing, "可用:", list(splits))
        return

    out = []
    t0 = time.time()
    for tag in want:
        for bw in bws:
            t1 = time.time()
            rows = run_split(splits[tag], args.floors, bw, args.dtheta,
                             args.dr, args.max_rx)
            if not rows:
                print(f"[{tag}] B={bw/1e6:.0f} MHz 无样本")
                continue
            a = agg(rows)
            a.update(split=tag, bw_mhz=bw / 1e6, floors=args.floors)
            out.append(a)
            print(f"[{tag:<26}] B={bw/1e6:>4.0f} MHz  n={a['n']:>5}  "
                  f"秩3(理想) {100*a['rank3_ideal']:>5.1f}%  "
                  f"受限后秩不足 {100*a['rank_short_both']:>5.2f}%  "
                  f"share_z 理想 {a['share_z_ideal_med']:.3f} -> 双重受限 "
                  f"{a['share_z_both_med']:.3f}  ({time.time()-t1:.1f}s)")

    keys = ["split", "bw_mhz", "floors", "n", "rank3_ideal", "rank_short_both",
            "share_z_ideal_med", "share_z_both_med", "d_eff_ideal_med",
            "d_eff_both_med", "kappa_med"]
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)
    print(f"\n汇总写入 {args.csv}  ({len(out)} 行)  总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
