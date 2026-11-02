#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_anchor_config.py -- 受控实验可行性侦察

目的：判断能否在**同一楼层（同一场景、同一材料、同一频段）内**，用
「锚点子采样」构造共面与非共面的受控对比，从而免去自研射线追踪引擎。

输出每个 split 抽查楼层的：
  1. 该楼层的唯一锚点数与逐个锚点坐标（尤其 z 高度分布）
  2. 按接收机坐标去重后，每个接收机位置被多少个不同锚点覆盖
     （只有被足够多锚点覆盖，子采样后才有链路可用）
  3. 覆盖层数分布（>=2 / >=3 / >=4 / >=5 锚点的接收机比例）
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402


SPLITS = [
    ("WiPTR", "train"),
    ("WiPTR", "val_grid"),
    ("WiPTR", "val_ood"),
    ("Wi3R", "grid"),
    ("Wi3R", "main_5tx"),
    ("Wi3R", "outofdistribution/fourrooms"),
]


def probe(info, n_floor=4):
    print("=" * 92)
    print(f"[{info.tag}]  {info.path.name}")
    with h5py.File(info.path, "r") as f:
        floors = wio.list_floors(f)
        print(f"  楼层总数: {len(floors)}")
        for fid, fidx in floors[:n_floor]:
            fc = wio.read_floor(f, fid)
            tx_uniq = np.unique(np.round(fc.tx_xyz, 4), axis=0)
            print(f"  -- floor {fid} (idx {fidx})  信道 {len(fc)}  唯一锚点 {tx_uniq.shape[0]}")
            for i, t in enumerate(tx_uniq):
                print(f"       anchor#{i}  x={t[0]:7.2f}  y={t[1]:7.2f}  z={t[2]:6.2f}")
            zs = np.unique(np.round(tx_uniq[:, 2], 3))
            print(f"     锚点 z 唯一值 {zs}   跨度 {np.ptp(tx_uniq[:, 2]):.3f} m")

            # 按接收机坐标去重
            rx_round = np.round(fc.rx_xyz, 3)
            _, inv = np.unique(rx_round, axis=0, return_inverse=True)
            tx_round = np.round(fc.tx_xyz, 4)
            tx_codes = np.ascontiguousarray(tx_round).view(
                [("x", "f8"), ("y", "f8"), ("z", "f8")]).ravel()
            # 每个接收机位置对应的锚点集合大小
            order = np.argsort(inv, kind="stable")
            inv_s = inv[order]
            tx_s = tx_codes[order]
            bounds = np.searchsorted(inv_s, np.arange(inv.max() + 1))
            ends = np.append(bounds[1:], inv_s.size)
            n_anchor = np.array([len(np.unique(tx_s[b:e])) for b, e in zip(bounds, ends)])
            print(f"     唯一接收机位置数 {n_anchor.size}  "
                  f"每位置被锚点数 min/med/max = "
                  f"{n_anchor.min()}/{int(np.median(n_anchor))}/{n_anchor.max()}")
            cov = " ".join(f">={k}:{(n_anchor >= k).mean():.3f}" for k in (2, 3, 4, 5))
            print(f"     覆盖比例 {cov}")


def main():
    root = Path(__file__).resolve().parent.parent / "data" / "public_raw"
    by_tag = {s.tag: s for s in wio.discover_splits(root)}
    for ds, sp in SPLITS:
        tag = f"{ds}/{sp}"
        if tag in by_tag:
            probe(by_tag[tag])
        else:
            print(f"[跳过] 未找到 {tag}")


if __name__ == "__main__":
    main()
