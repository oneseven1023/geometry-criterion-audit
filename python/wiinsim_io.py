#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wiinsim_io.py -- Qualcomm Wireless Indoor Simulations (Wi3R / WiPTR) 统一读取层

把 ``dataset_sparse.h5`` 的稀疏存储解析为「一个场景 = 一个楼层平面图」的
链路级记录。所有下游脚本（normalize / baseline / eval / analysis）共用本模块，
不重复实现解析逻辑。

HDF5 结构（已由 inspect_data.py 实测确认）
------------------------------------------
====================  ==============  ==========================================
数据集                 形状             含义
====================  ==============  ==========================================
``tx``                (N_tx, 3)       发射机 / 锚点坐标，米
``rx``                (N_rx, 3)       接收机坐标，米
``channels``          (M, 3)          (tx_idx, rx_idx, first_path_idx)
``paths``             (K, 7)          逐路径 MPC
``floor_idx_channels``(N_floor, 2)    每楼层 (floor_idx, channels 起始索引)
====================  ==============  ==========================================

``paths`` 七列依次为
``(gains_mag, gains_phase, delay_ns, aod_az_rad, zod_rad, aoa_az_rad, zoa_rad)``。
第 ``m`` 条信道的路径区间为 ``paths[channels[m,2] : channels[m+1,2]]``，左闭右开；
同一楼层内的路径块在 ``paths`` 中是连续的，因此可以整块读取后用
``np.add.reduceat`` 向量化聚合。

功率口径（关键，已做物理校验）
------------------------------
官方实现 ``wiinsim.utils.get_total_power_received_from_mpc`` 定义为

    总路径增益[dB] = 10 * log10( sum_i 10 ** (gains_mag_i / 5) )

即单径线性增益 ``w_i = 10 ** (gains_mag_i / 5)``。本模块沿用该口径。
校验依据（见 docs/data_schema.md）：以该口径聚合的总功率对 ``log10(d)``
回归得到 20.7 dB/十倍程的斜率，与自由空间一致；改用 ``/10`` 口径则整体抬高
约 33.6 dB，在 1 m 处得到 -8.8 dB 的无量纲路径增益，物理上不成立。

时延口径
--------
``delay_ns`` 为绝对传播时延（不是相对首径的过剩时延）。实测 WiPTR/val_ood 中
97.7% 的信道满足 ``min(delay)*c ≈ ||tx - rx||``，即最先到达径与几何直视距离一致。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

C_M_PER_NS = 0.299792458  # 光速，m/ns

PATH_COLUMNS = (
    "gains_mag",
    "gains_phase",
    "delay_ns",
    "aod_az_rad",
    "zod_rad",
    "aoa_az_rad",
    "zoa_rad",
)

# 功率口径：单径线性增益 = 10 ** (gains_mag / GAIN_DB_DIVISOR)
GAIN_DB_DIVISOR = 5.0

# 判定视距的超额路径长度阈值，米
LOS_EXCESS_M = 0.20


@dataclass(frozen=True)
class SplitInfo:
    """一个 h5 文件对应的数据集与划分。"""

    dataset: str  # Wi3R 或 WiPTR
    split: str  # main_5tx / grid / train / val / test / ...
    path: Path

    @property
    def tag(self) -> str:
        return f"{self.dataset}/{self.split}"


def discover_splits(root: str | Path) -> list[SplitInfo]:
    """在 ``root`` 下递归寻找 dataset_sparse.h5，并推断 dataset / split 名称。

    期望的目录形如 ``<root>/Wi3R/main_5tx/dataset_sparse.h5``，也兼容
    ``<root>/Wi3R/outofdistribution/fourrooms/dataset_sparse.h5`` 这种多级路径。
    """
    root = Path(root)
    out: list[SplitInfo] = []
    for p in sorted(root.rglob("dataset_sparse.h5")):
        rel = p.relative_to(root)
        parts = rel.parts[:-1]  # 去掉文件名
        if not parts:
            continue
        dataset = parts[0]
        split = "/".join(parts[1:]) if len(parts) > 1 else "root"
        out.append(SplitInfo(dataset=dataset, split=split, path=p))
    return out


def floor_table(f: h5py.File) -> np.ndarray:
    """返回 (N_floor, 2) 的 [floor_idx, first_channel_idx] 表。"""
    return np.asarray(f["floor_idx_channels"][:], dtype=np.int64)


def floor_channel_bounds(f: h5py.File) -> np.ndarray:
    """返回 (N_floor+1,) 的 channels 分界索引，左闭右开。

    即第 i 个楼层占用 ``channels[bounds[i] : bounds[i+1]]``。
    """
    tbl = floor_table(f)
    bounds = np.append(tbl[:, 1], np.asarray(f["channels"].shape[0], dtype=np.int64))
    return bounds


def list_floors(f: h5py.File) -> list[tuple[int, int]]:
    """返回 [(floor_id, floor_idx), ...]。floor_id 为行号，floor_idx 为文件名编号。"""
    tbl = floor_table(f)
    return [(i, int(tbl[i, 0])) for i in range(tbl.shape[0])]


@dataclass
class FloorChannels:
    """一个楼层（场景）的全部链路记录，字段均为数组，长度等于信道数。

    Attributes
    ----------
    floor_id, floor_idx : 楼层行号与文件编号
    tx_idx, rx_idx      : 在 h5 全局数组中的索引
    tx_xyz, rx_xyz      : (M, 3) 收发机坐标，米
    rss_db              : (M,) 官方口径总路径增益，dB
    tof_first_ns        : (M,) 最先到达径时延，ns
    tof_strong_ns       : (M,) 最强径时延，ns
    path_count          : (M,) 有效路径数
    delay_spread_ns     : (M,) 功率加权 RMS 时延扩展，ns
    aoa_az_deg          : (M,) 最强径到达方位角，度
    aoa_el_deg          : (M,) 最强径到达俯仰角，度
    aod_az_deg          : (M,) 最强径离开方位角，度
    aod_el_deg          : (M,) 最强径离开俯仰角，度
    los_flag            : (M,) 是否视距，1 或 0
    d_geo_m             : (M,) 收发机几何直线距离，米
    rx_pos_id           : (M,) 每个锚点下的接收机位置编号
    """

    floor_id: int
    floor_idx: int
    tx_idx: np.ndarray
    rx_idx: np.ndarray
    tx_xyz: np.ndarray
    rx_xyz: np.ndarray
    rss_db: np.ndarray
    tof_first_ns: np.ndarray
    tof_strong_ns: np.ndarray
    path_count: np.ndarray
    delay_spread_ns: np.ndarray
    aoa_az_deg: np.ndarray
    aoa_el_deg: np.ndarray
    aod_az_deg: np.ndarray
    aod_el_deg: np.ndarray
    los_flag: np.ndarray
    d_geo_m: np.ndarray
    rx_pos_id: np.ndarray

    def __len__(self) -> int:
        return int(self.tx_idx.shape[0])


def _reduceat_first_max(g: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """对每个分段求「最大值首次出现的下标」（相对全数组的下标）。

    等价于逐段 ``np.argmax``，但完全向量化，避免百万级信道上的 Python 循环。
    """
    seg_max = np.maximum.reduceat(g, starts)
    seg_max_rep = np.repeat(seg_max, counts)
    pos = np.arange(g.size, dtype=np.int64)
    big = np.iinfo(np.int64).max
    first = np.minimum.reduceat(np.where(g >= seg_max_rep, pos, big), starts)
    return first


def read_floor(f: h5py.File, floor_id: int) -> FloorChannels:
    """读取单个楼层的全部信道，并做向量化聚合。"""
    tbl = floor_table(f)
    bounds = floor_channel_bounds(f)
    floor_idx = int(tbl[floor_id, 0])

    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    if ch.size == 0:
        raise ValueError(f"楼层 {floor_id} (floor_idx={floor_idx}) 没有信道")

    # 只读取被本楼层引用到的行，避免在百万级 rx 表上做整体读取
    tx_idx = ch[:, 0]
    rx_idx = ch[:, 1]
    pstart = ch[:, 2]

    utx = np.unique(tx_idx)
    urx = np.unique(rx_idx)
    tx_tab = np.asarray(f["tx"][utx], dtype=np.float64)
    rx_tab = np.asarray(f["rx"][urx], dtype=np.float64)
    tx_map = np.searchsorted(utx, tx_idx)
    rx_map = np.searchsorted(urx, rx_idx)

    # 该楼层的路径块在 paths 中是连续的
    p0 = int(pstart[0])
    # 末条信道的路径块终点：若还有下一条信道则取其起点，否则取文件末尾
    if c1 < np.asarray(f["channels"].shape[0]):
        p_end = int(np.asarray(f["channels"][c1, 2]))
    else:
        p_end = int(np.asarray(f["paths"].shape[0]))
    paths = np.asarray(f["paths"][p0:p_end], dtype=np.float64)
    rel_starts = pstart - p0
    counts = np.diff(np.append(rel_starts, paths.shape[0])).astype(np.int64)

    gains, phase, delay = paths[:, 0], paths[:, 1], paths[:, 2]
    aod_az, zod, aoa_az, zoa = paths[:, 3], paths[:, 4], paths[:, 5], paths[:, 6]

    # 功率加权聚合
    w = 10.0 ** (gains / GAIN_DB_DIVISOR)
    w_sum = np.add.reduceat(w, rel_starts)
    wr_sum = np.add.reduceat(w * delay, rel_starts)
    wr2_sum = np.add.reduceat(w * delay * delay, rel_starts)
    mean_tau = wr_sum / w_sum
    var_tau = np.maximum(wr2_sum / w_sum - mean_tau * mean_tau, 0.0)

    tof_first = np.minimum.reduceat(delay, rel_starts)
    tof_last = np.maximum.reduceat(delay, rel_starts)

    strong_pos = _reduceat_first_max(gains, rel_starts, counts)
    tof_strong = delay[strong_pos]

    tx_xyz = tx_tab[tx_map]
    rx_xyz = rx_tab[rx_map]
    d_geo = np.linalg.norm(tx_xyz - rx_xyz, axis=1)

    excess = tof_first * C_M_PER_NS - d_geo
    los = (excess <= LOS_EXCESS_M).astype(np.int8)

    # 每个锚点下的接收机位置编号：raw rx_idx 对「每锚点信道数」取模
    n_rx_per_tx = int(np.bincount(tx_idx).max())
    rx_pos_id = rx_idx % n_rx_per_tx

    # 天顶角转俯仰角
    el_aoa = 90.0 - np.degrees(zoa[strong_pos])
    el_aod = 90.0 - np.degrees(zod[strong_pos])

    return FloorChannels(
        floor_id=int(floor_id),
        floor_idx=floor_idx,
        tx_idx=tx_idx,
        rx_idx=rx_idx,
        tx_xyz=tx_xyz,
        rx_xyz=rx_xyz,
        rss_db=10.0 * np.log10(w_sum),
        tof_first_ns=tof_first,
        tof_strong_ns=tof_strong,
        path_count=counts,
        delay_spread_ns=np.sqrt(var_tau),
        aoa_az_deg=np.degrees(aoa_az[strong_pos]),
        aoa_el_deg=el_aoa,
        aod_az_deg=np.degrees(aod_az[strong_pos]),
        aod_el_deg=el_aod,
        los_flag=los,
        d_geo_m=d_geo,
        rx_pos_id=rx_pos_id,
    )


def read_floor_paths(f: h5py.File, floor_id: int) -> tuple[np.ndarray, np.ndarray]:
    """读取单楼层的逐路径原始块。

    Returns
    -------
    paths : (K, 7) float64，列定义见 ``PATH_COLUMNS``
    channel_of_path : (K,) int64，每条路径所属信道在该楼层内的局部编号
    """
    bounds = floor_channel_bounds(f)
    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    p0 = int(ch[0, 2])
    if c1 < np.asarray(f["channels"].shape[0]):
        p_end = int(np.asarray(f["channels"][c1, 2]))
    else:
        p_end = int(np.asarray(f["paths"].shape[0]))
    paths = np.asarray(f["paths"][p0:p_end], dtype=np.float64)
    rel = ch[:, 2] - p0
    counts = np.diff(np.append(rel, paths.shape[0]))
    channel_of_path = np.repeat(np.arange(ch.shape[0], dtype=np.int64), counts)
    return paths, channel_of_path


def iter_floors(info: SplitInfo, floor_ids: list[int] | None = None,
                limit: int | None = None) -> Iterator[tuple[h5py.File, int]]:
    """按楼层迭代，惰性打开文件。

    每次 yield ``(打开的 h5 文件句柄, floor_id)``，由调用方在使用完该楼层后
    继续迭代。文件句柄在迭代结束时关闭。
    """
    with h5py.File(info.path, "r") as f:
        ids = [fid for fid, _ in list_floors(f)]
        if floor_ids is not None:
            ids = [i for i in ids if i in set(floor_ids)]
        if limit is not None:
            ids = ids[:limit]
        for fid in ids:
            yield f, fid
