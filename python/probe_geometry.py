#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_geometry.py -- C2 最小实验前的几何侦察

目的：确认 WiPTR 各 split 的楼层几何与多锚点结构，为「判据失效审计」确定
分层维度（锚点数、共面程度、LoS 比例、每信道路径数）。

输出每个 split 前若干楼层的：锚点数、接收机数、每锚点接收机数、
收发机 z 的离散程度（是否共面）、rx_pos_id 是否跨锚点坐标一致、
每信道路径数分布。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402


SPLITS = [
    ("WiPTR", "val_grid"),
    ("WiPTR", "train_grid"),
    ("WiPTR", "val_ood"),
    ("Wi3R", "main_5tx"),
]


def probe(info, n_floor=3):
    print("=" * 78)
    print(f"[{info.tag}]  {info.path}")
    with h5py.File(info.path, "r") as f:
        floors = wio.list_floors(f)
        print(f"  楼层总数: {len(floors)}")
        for fid, fidx in floors[:n_floor]:
            fc = wio.read_floor(f, fid)
            tx_idx = np.unique(fc.tx_idx)
            # 每锚点信道数
            cnt = np.bincount(np.searchsorted(tx_idx, fc.tx_idx), minlength=tx_idx.size)
            txz = fc.tx_xyz[:, 2]
            rxz = fc.rx_xyz[:, 2]
            print(f"  floor {fid} (idx={fidx}): 信道 {len(fc)}  锚点 {tx_idx.size}  "
                  f"接收机 {np.unique(fc.rx_idx).size}")
            print(f"      每锚点信道数 min/med/max = {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}")
            print(f"      tx z: 唯一值 {np.unique(np.round(txz, 3)).size}  "
                  f"范围 {txz.min():.2f}~{txz.max():.2f}  (跨度 {np.ptp(txz):.3f} m)")
            print(f"      rx z: 唯一值 {np.unique(np.round(rxz, 3)).size}  "
                  f"范围 {rxz.min():.2f}~{rxz.max():.2f}  (跨度 {np.ptp(rxz):.3f} m)")
            print(f"      tx x 跨度 {np.ptp(fc.tx_xyz[:,0]):.2f} m, y 跨度 {np.ptp(fc.tx_xyz[:,1]):.2f} m")
            print(f"      rx x 跨度 {np.ptp(fc.rx_xyz[:,0]):.2f} m, y 跨度 {np.ptp(fc.rx_xyz[:,1]):.2f} m")
            # 每信道路径数
            print(f"      path_count: min/med/max = "
                  f"{fc.path_count.min()}/{int(np.median(fc.path_count))}/{fc.path_count.max()}")
            print(f"      LoS 比例 {fc.los_flag.mean():.3f}  "
                  f"时延扩展 med {np.median(fc.delay_spread_ns):.1f} ns")
            # rx_pos_id 是否跨锚点一致：按 rx_pos_id 分组，看 rx_xyz 是否唯一
            for pid in np.unique(fc.rx_pos_id)[:3]:
                m = fc.rx_pos_id == pid
                xyz = fc.rx_xyz[m]
                uniq = np.unique(np.round(xyz, 4), axis=0)
                print(f"      rx_pos_id={pid}: 出现 {m.sum()} 次(跨锚点), "
                      f"坐标唯一解 {uniq.shape[0]}  -> "
                      f"{'一致' if uniq.shape[0] == 1 else '不一致'}")


def main():
    root = Path(__file__).resolve().parent.parent / "data" / "public_raw"
    splits = wio.discover_splits(root)
    by_tag = {s.tag: s for s in splits}
    for ds, sp in SPLITS:
        tag = f"{ds}/{sp}"
        if tag in by_tag:
            probe(by_tag[tag])
        else:
            print(f"[跳过] 未找到 {tag}")


if __name__ == "__main__":
    main()
