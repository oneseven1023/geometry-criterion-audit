#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspect_data.py -- Qualcomm Wireless Indoor Simulations (Wi3R / WiPTR) HDF5 结构探查

用途
    在不加载全部数据的前提下，打印 ``dataset_sparse.h5`` 的数据集清单、形状、
    数值范围与若干行样例，用于确认收发机坐标、稀疏信道索引与多径字段的实际语义。

背景（来源：官方仓库 Qualcomm-AI-research/WiInSim, src/wiinsim/dataset.py）
    每个 split 是一个 HDF5，包含四个核心数据集：

    ================  ===============  ==========================================
    数据集             形状              含义
    ================  ===============  ==========================================
    ``tx``            (N_tx, 3)        发射机 / 锚点坐标，米
    ``rx``            (N_rx, 3)        接收机坐标，米
    ``channels``      (M, 3)           (tx_idx, rx_idx, first_path_idx) 稀疏索引
    ``paths``         (K, 7)           逐路径多径分量，见下
    ================  ===============  ==========================================

    ``paths`` 七列依次为
    ``(gains_mag, gains_phase, delay_ns, aod_az_rad, zod_rad, aoa_az_rad, zoa_rad)``。
    第 ``m`` 条信道的路径区间为 ``paths[first_path_idx : next_first_path_idx]``，
    即左闭右开；最后一条信道取到文件末尾。

    另有一个 ``floor_idx_channels`` 数据集，形状 (N_floor, 2)，
    每行为某楼层在 ``channels`` 中的起止索引。

运行
    python python/inspect_data.py --h5 <path/to/dataset_sparse.h5> [--rows 3]
    python python/inspect_data.py --scan data/public_raw   # 扫描目录下全部 h5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PATH_COLUMNS = [
    "gains_mag",
    "gains_phase",
    "delay_ns",
    "aod_az_rad",
    "zod_rad",
    "aoa_az_rad",
    "zoa_rad",
]


def _fmt_array(a: np.ndarray, rows: int = 3) -> str:
    """把数组的前若干行渲染成紧凑字符串。"""
    a = np.asarray(a)
    head = a[:rows]
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        return f"shape={a.shape} dtype={a.dtype}\n{head}"


def inspect_one(h5_path: Path, rows: int = 3) -> None:
    print("=" * 78)
    print(f"文件: {h5_path}")
    print(f"大小: {h5_path.stat().st_size / 1e6:.1f} MB")
    print("=" * 78)

    with h5py.File(h5_path, "r") as f:
        print("-- 顶层数据集 --")
        for k in f.keys():
            ds = f[k]
            print(f"   {k:22s} shape={str(ds.shape):18s} dtype={ds.dtype}")

        # ---- 坐标 ----
        tx = f["tx"][:]
        rx = f["rx"][:]
        print(f"\n-- 收发机坐标 --")
        print(f"   tx: {_fmt_array(tx, rows)}")
        print(f"   rx: {_fmt_array(rx, rows)}")
        print(f"   tx 坐标范围 x[{tx[:,0].min():.2f},{tx[:,0].max():.2f}] "
              f"y[{tx[:,1].min():.2f},{tx[:,1].max():.2f}] "
              f"z[{tx[:,2].min():.2f},{tx[:,2].max():.2f}]")
        print(f"   rx 坐标范围 x[{rx[:,0].min():.2f},{rx[:,0].max():.2f}] "
              f"y[{rx[:,1].min():.2f},{rx[:,1].max():.2f}] "
              f"z[{rx[:,2].min():.2f},{rx[:,2].max():.2f}]")

        # ---- 稀疏信道索引 ----
        ch = f["channels"]
        print(f"\n-- channels (稀疏索引) --")
        print(f"   {_fmt_array(ch[:10], rows=5)}")
        n_ch = ch.shape[0]
        tx_ids = np.unique(np.asarray(ch[:, 0]))
        rx_ids = np.unique(np.asarray(ch[:, 1]))
        print(f"   信道总数 {n_ch}")
        print(f"   引用到的 tx 索引数 {tx_ids.size}（取值范围 {tx_ids.min()}~{tx_ids.max()}）")
        print(f"   引用到的 rx 索引数 {rx_ids.size}（取值范围 {rx_ids.min()}~{rx_ids.max()}）")
        print(f"   tx 数组长度 {tx.shape[0]}，rx 数组长度 {rx.shape[0]}")

        # 每个 tx 对应的 rx 数（判断锚点均匀性）
        cnt = {}
        for t in np.asarray(ch[:, 0]):
            cnt[t] = cnt.get(t, 0) + 1
        counts = np.array(sorted(cnt.values()))
        print(f"   每锚点信道数: min={counts.min()} max={counts.max()} "
              f"中位={int(np.median(counts))}")

        # ---- 路径 ----
        pth = f["paths"]
        print(f"\n-- paths (逐路径 MPC, 7 列) --")
        print(f"   列定义: {PATH_COLUMNS}")
        print(f"   {_fmt_array(pth[:8], rows=8)}")
        print(f"   路径总数 {pth.shape[0]}")

        # 第 0 条信道的路径区间
        first = int(ch[0][2])
        last = int(ch[1][2]) if n_ch > 1 else pth.shape[0]
        seg = pth[first:last]
        print(f"\n   第 0 条信道: first_path_idx={first}, 路径数={last - first}")
        print(f"   该信道路径数分布: 全文件最大路径数="
              f"{max(int(np.asarray(ch[i+1][2])) - int(np.asarray(ch[i][2])) for i in range(min(n_ch-1, 2000)))}"
              f" (前 2000 条信道内)")

        # 数值范围诊断（判断 gains_mag 是 dB 还是线性）
        g = np.asarray(pth[:, 0], dtype=np.float64)
        print(f"\n-- 第 0 列 gains_mag 数值范围 --")
        print(f"   min={g.min():.3f} max={g.max():.3f} mean={g.mean():.3f} "
              f"中位={np.median(g):.3f}")
        print(f"   负值占比 {np.mean(g < 0)*100:.1f}%   为 0 的占比 {np.mean(g == 0)*100:.1f}%")
        d = np.asarray(pth[:, 2], dtype=np.float64)
        print(f"-- 第 2 列 delay_ns 数值范围 --")
        print(f"   min={d.min():.3f} ns max={d.max():.3f} ns 中位={np.median(d):.3f} ns")

        # ---- 楼层索引 ----
        if "floor_idx_channels" in f:
            fic = f["floor_idx_channels"]
            print(f"\n-- floor_idx_channels --")
            print(f"   shape={fic.shape} dtype={fic.dtype}")
            print(f"   前 5 行:\n{np.asarray(fic[:5])}")
            print(f"   楼层数（含网格数据的额外划分）={fic.shape[0]}")
            # floor_idx 与 channels 索引的对应关系
            print(f"   说明: 第 i 行的第二个值是 channels 的起始索引（左闭右开的下一个边界）")

        # ---- 其它属性 ----
        print(f"\n-- 文件级属性 --")
        for k, v in f.attrs.items():
            print(f"   {k} = {v}")


def scan(root: Path, rows: int = 3) -> None:
    files = sorted(root.rglob("dataset_sparse.h5"))
    if not files:
        print(f"未在 {root} 下找到 dataset_sparse.h5", file=sys.stderr)
        return
    print(f"共发现 {len(files)} 个 h5 文件\n")
    for p in files:
        inspect_one(p, rows=rows)
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--h5", type=Path, help="单个 dataset_sparse.h5 路径")
    g.add_argument("--scan", type=Path, help="递归扫描该目录下所有 dataset_sparse.h5")
    ap.add_argument("--rows", type=int, default=3, help="样例打印行数")
    a = ap.parse_args()

    if a.h5:
        if not a.h5.exists():
            print(f"文件不存在: {a.h5}", file=sys.stderr)
            return 1
        inspect_one(a.h5, rows=a.rows)
    else:
        scan(a.scan, rows=a.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
