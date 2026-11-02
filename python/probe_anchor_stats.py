#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_anchor_stats.py -- 各 split 的锚点构型统计

对每个 split 抽样若干楼层，统计：
  n_anchor        唯一锚点数
  z_span          锚点 z 跨度（越大越不共面）
  n_anchor_uniq_z 锚点 z 的唯一值个数（1 表示严格共面）
  rx_cov_med      每个接收机位置被锚点覆盖数的中位
并给出分位数与「严格共面楼层」占比。用来判定：
  1. 此前「WiPTR 锚点全部共面于 2.30 m」是否成立
  2. 能否在同一楼层内用锚点子采样构造受控对比
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

N_FLOOR = 80


def stats(info):
    rec = {"n_anchor": [], "z_span": [], "uniq_z": [], "rx_cov": []}
    with h5py.File(info.path, "r") as f:
        floors = wio.list_floors(f)
        step = max(1, len(floors) // N_FLOOR)
        for fid, _ in floors[::step][:N_FLOOR]:
            fc = wio.read_floor(f, fid)
            tx = np.unique(np.round(fc.tx_xyz, 4), axis=0)
            rec["n_anchor"].append(tx.shape[0])
            rec["z_span"].append(float(np.ptp(tx[:, 2])))
            rec["uniq_z"].append(int(np.unique(np.round(tx[:, 2], 3)).size))
            rx = np.round(fc.rx_xyz, 3)
            _, inv = np.unique(rx, axis=0, return_inverse=True)
            tx_codes = np.ascontiguousarray(np.round(fc.tx_xyz, 4)).view(
                [("x", "f8"), ("y", "f8"), ("z", "f8")]).ravel()
            order = np.argsort(inv, kind="stable")
            inv_s, tx_s = inv[order], tx_codes[order]
            b = np.searchsorted(inv_s, np.arange(inv.max() + 1))
            e = np.append(b[1:], inv_s.size)
            cov = np.array([len(np.unique(tx_s[i:j])) for i, j in zip(b, e)])
            rec["rx_cov"].append(float(np.median(cov)))
    return rec


def q(a, p):
    return float(np.percentile(a, p))


def main():
    root = Path(__file__).resolve().parent.parent / "data" / "public_raw"
    splits = wio.discover_splits(root)
    print(f"{'split':34s} {'n_floor':>7s} {'锚点数中位':>9s} {'z跨度中位':>9s} "
          f"{'z跨度90%':>9s} {'严格共面%':>9s} {'覆盖中位':>9s}")
    print("-" * 92)
    for info in splits:
        try:
            r = stats(info)
        except Exception as ex:  # noqa: BLE001
            print(f"{info.tag:34s}  读取失败 {ex}")
            continue
        na = np.array(r["n_anchor"], dtype=float)
        zs = np.array(r["z_span"], dtype=float)
        uz = np.array(r["uniq_z"], dtype=float)
        cov = np.array(r["rx_cov"], dtype=float)
        print(f"{info.tag:34s} {na.size:7d} {np.median(na):9.1f} {np.median(zs):9.3f} "
              f"{q(zs, 90):9.3f} {(uz == 1).mean() * 100:8.1f}% {np.median(cov):9.1f}")


if __name__ == "__main__":
    main()
