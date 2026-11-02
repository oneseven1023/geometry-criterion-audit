#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_driver_probe.py -- 垂直退化的驱动因素定位

层内配对检验（c2_controllability_probe.py）显示：在 WiPTR 内部，楼层锚点是否
共面几乎不改变 share_z（严格共面 0.823，明显非共面 0.865）。因此「共面锚点导致
垂直退化」这一归因不成立，WiPTR 与 Wi3R 的差距另有来源。

本脚本按接收机高度 rx_z 与「接收机相对锚点的垂向距离」分档，比较 share_z 与
虚拟锚点方向的垂直分量 mean|u_z|，用来定位真正的驱动因素。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio            # noqa: E402
import c2_fullscale_vec as c2v      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
N_FLOOR = 200

SPLITS = [("WiPTR", "train"), ("WiPTR", "val"), ("Wi3R", "main_5tx")]


def collect(info):
    rx_z, uz, sz, dz = [], [], [], []
    with h5py.File(info.path, "r") as f:
        floors = wio.list_floors(f)
        step = max(1, len(floors) // N_FLOOR)
        for fid, _ in floors[::step][:N_FLOOR]:
            r = c2v.process_floor(f, fid, 30.0)
            if r is None:
                continue
            bounds = wio.floor_channel_bounds(f)
            c0, c1 = int(bounds[fid]), int(bounds[fid + 1])
            ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
            utx = np.unique(ch[:, 0])
            atz = float(np.mean(np.asarray(f["tx"][utx], dtype=np.float64)[:, 2]))
            rx_z.append(r["rx_z"])
            uz.append(r["mean_abs_uz"])
            sz.append(r["share_z"])
            dz.append(np.abs(r["rx_z"] - atz))
    return (np.concatenate(rx_z), np.concatenate(uz),
            np.concatenate(sz), np.concatenate(dz))


def binned(v, sz, uz, edges, labels):
    print(f"    {'档':22s} {'链路':>8s} {'share_z中位':>11s} {'sz>0.5%':>8s} "
          f"{'mean|uz|中位':>12s}")
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        m = (v >= lo) & (v < hi)
        if m.sum() == 0:
            continue
        print(f"    {lab:22s} {m.sum():8d} {np.median(sz[m]):11.4f} "
              f"{(sz[m] > 0.5).mean() * 100:7.1f}% {np.median(uz[m]):12.4f}")


def main():
    for ds, sp in SPLITS:
        info = next((s for s in wio.discover_splits(ROOT) if s.tag == f"{ds}/{sp}"), None)
        if info is None:
            continue
        rx_z, uz, sz, dz = collect(info)
        print("=" * 90)
        print(f"== {ds}/{sp} ==  链路 {sz.size}   rx_z 范围 {rx_z.min():.2f}~{rx_z.max():.2f}  "
              f"share_z 中位 {np.median(sz):.4f}  mean|uz| 中位 {np.median(uz):.4f}")
        print("  按 rx_z 分档:")
        q = np.percentile(rx_z, [0, 25, 50, 75, 100])
        binned(rx_z, sz, uz, q, [f"[{q[i]:.2f},{q[i+1]:.2f})" for i in range(4)])
        print("  按 |rx_z - 锚点平均 z| 分档:")
        q2 = np.percentile(dz, [0, 25, 50, 75, 100])
        binned(dz, sz, uz, q2, [f"[{q2[i]:.2f},{q2[i+1]:.2f})" for i in range(4)])
        print(f"  corr(share_z, mean|uz|) = {np.corrcoef(sz, uz)[0,1]:.4f}   "
              f"corr(share_z, |rx_z-anchor_z|) = {np.corrcoef(sz, dz)[0,1]:.4f}")


if __name__ == "__main__":
    main()
