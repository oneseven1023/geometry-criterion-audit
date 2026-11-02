#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_controllability_probe.py -- 层内配对检验（受控实验可行性验证）

动机：此前论文以「WiPTR 为共面数据集、Wi3R 为非共面数据集」做跨数据集对照。
但锚点构型统计显示 WiPTR/train 仅约 47.5% 的楼层严格共面，其余楼层锚点存在
高度差。因此跨数据集对照混入了数据集（场景尺寸、材料、频段）差异。

本脚本把「共面性」降为**同一 split 内的楼层级变量**：按楼层锚点 z 跨度分档，
在同一 split 内比较垂直方差份额 share_z、条件数 kappa、秩。若分档间差异显著，
则共面性是层的层内可控变量，受控实验可在现有数据上完成，无需自研射线追踪。

口径与 c2_fullscale_vec.process_floor 完全一致（判据侧，动态范围 30 dB）。
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

SPLITS = [
    ("WiPTR", "train"),
    ("WiPTR", "val"),
    ("WiPTR", "test"),
    ("WiPTR", "val_ood"),
    ("Wi3R", "main_5tx"),
]

# 按楼层锚点 z 跨度分档
BINS = [(0.0, 0.0, "严格共面 zs=0"),
        (1e-9, 0.3, "近共面 0<zs<=0.3"),
        (0.3, 1.0, "中等 0.3<zs<=1.0"),
        (1.0, 1e9, "明显非共面 zs>1.0")]


def bin_of(zs):
    for lo, hi, name in BINS:
        if lo - 1e-12 <= zs <= hi:
            return name
    return BINS[-1][2]


def run_split(info):
    rows = []  # (z_span, share_z array)
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
            txz = np.asarray(f["tx"][utx], dtype=np.float64)[:, 2]
            rows.append((float(np.ptp(txz)), r))
    return rows


def main():
    print(f"{'分组':26s} {'楼层':>5s} {'链路':>8s} {'share_z中位':>11s} "
          f"{'sz>0.5%':>8s} {'d_eff中位':>9s} {'kappa中位':>9s} {'秩3%':>6s}")
    print("-" * 92)
    for ds, sp in SPLITS:
        info = next((s for s in wio.discover_splits(ROOT) if s.tag == f"{ds}/{sp}"), None)
        if info is None:
            print(f"[跳过] {ds}/{sp}")
            continue
        rows = run_split(info)
        if not rows:
            continue
        print(f"== {ds}/{sp} ==")
        for _, _, name in BINS:
            sel = [r for zs, r in rows if bin_of(zs) == name]
            if not sel:
                continue
            sz = np.concatenate([r["share_z"] for r in sel])
            de = np.concatenate([r["d_eff"] for r in sel])
            ka = np.concatenate([r["kappa"] for r in sel])
            rk = np.concatenate([r["rank"] for r in sel])
            print(f"{name:26s} {len(sel):5d} {sz.size:8d} {np.median(sz):11.4f} "
                  f"{(sz > 0.5).mean() * 100:7.1f}% {np.median(de):9.3f} "
                  f"{np.median(ka):9.2f} {(rk == 3).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
