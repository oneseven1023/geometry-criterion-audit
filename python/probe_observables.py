#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_observables.py -- 侦察每条 MPC 可观测的物理量，判断观测模态剥夺是否可做。

适用条件审计的第一个前置条件是「到达方向可观测」，它要求至少一端有阵列。
本脚本统计 paths 七列（gain_mag, gain_phase, delay_ns, aod_az, zod, aoa_az, zoa）
在真实数据里的有效性与取值分布，确认能否用「丢弃 AoD 或 AoA」来模拟模态剥夺。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"


def report(tag: str, path: Path, floors: int = 2):
    print(f"\n===== {tag} =====")
    with h5py.File(path, "r") as f:
        ids = [fid for fid, _ in wio.list_floors(f)][:floors]
        for fid in ids:
            paths, ch_of_path = wio.read_floor_paths(f, fid)
            k = paths.shape[0]
            names = wio.PATH_COLUMNS
            finite = np.isfinite(paths)
            print(f"  floor {fid}: paths {k}, 信道 {ch_of_path.max()+1}")
            for j, nm in enumerate(names):
                col = paths[:, j]
                nz = np.mean(np.abs(col) > 1e-12)
                print(f"    {nm:>12}: 有限 {finite[:, j].mean():.4f}  非零 {nz:.4f}  "
                      f"min {col.min():+.4f} max {col.max():+.4f} 中位 {np.median(col):+.4f}")
            # 到达/离开天顶角是否落在 (0, pi) 内（真角）还是恒 0/pi（缺列）
            zod, zoa = paths[:, 4], paths[:, 6]
            print(f"    zod 落在(0,pi)比例 {(np.logical_and(zod>0, zod<np.pi)).mean():.4f}  "
                  f"zoa 落在(0,pi)比例 {(np.logical_and(zoa>0, zoa<np.pi)).mean():.4f}")
            # 每信道锚点数：本 split 是否真多锚点
            break


def main():
    splits = {s.tag: s for s in wio.discover_splits(ROOT)}
    for tag in ["WiPTR/val_grid", "WiPTR/val_ood", "Wi3R/main_5tx"]:
        if tag in splits:
            report(tag, splits[tag].path)


if __name__ == "__main__":
    main()
