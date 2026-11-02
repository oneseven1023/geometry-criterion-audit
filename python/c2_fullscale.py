#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_fullscale.py -- 在全部公开射线上跑 C2 判据失效审计（系统级贡献的证据主体）。

对 data/public_raw 下每一个 dataset_sparse.h5，遍历全部楼层与全部接收机，
按接收机真实坐标分组统计判据侧与真实侧的指标，输出
results/c2full_<dataset>_<split>.csv。

用法
----
python c2_fullscale.py                      # 全部 split，B=100 MHz
python c2_fullscale.py --bw 20e6            # 低带宽
python c2_fullscale.py --only WiPTR/val_grid
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
import c2_criterion_audit as aud  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=100e6)
    ap.add_argument("--dtheta", type=float, default=20.0)
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--only", default=None, help="只跑指定 split，如 WiPTR/val_grid")
    ap.add_argument("--tag-suffix", default="", help="输出文件名后缀，如 _20MHz")
    args = ap.parse_args()

    splits = wio.discover_splits(ROOT)
    if args.only:
        splits = [s for s in splits if s.tag == args.only]
    cfg = dict(bw_hz=args.bw, dtheta_deg=args.dtheta, dr_db=args.dr)

    print(f"全量审计：{len(splits)} 个 split，B={args.bw/1e6:.0f} MHz "
          f"dtheta={args.dtheta:.0f}deg DR={args.dr:.0f}dB")
    grand = []
    t0 = time.time()
    for i, info in enumerate(splits, 1):
        t1 = time.time()
        rows = aud.run_split(info, None, args.bw, args.dtheta, args.dr, None)
        aud.summarize(rows, info.tag, cfg)
        fname = f"c2full_{info.tag.replace('/', '_')}{args.tag_suffix}.csv"
        write_csv(rows, RES / fname)
        print(f"  -> {fname}  ({len(rows)} 行，用时 {time.time()-t1:.1f}s)")
        grand.append((info.tag, len(rows)))
    print(f"\n全部完成，总用时 {time.time()-t0:.1f}s")
    print("总样本（按接收机真实坐标分组的链路数）：", sum(n for _, n in grand))
    for tag, n in grand:
        print(f"  {tag:<32}{n:>7}")


if __name__ == "__main__":
    main()
