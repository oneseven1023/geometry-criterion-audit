#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_fullscale_vec.py -- C2 判据失效审计的全量向量化实现（系统级贡献证据主体）

与 c2_criterion_audit.py 的关系
-------------------------------
c2_criterion_audit.py 逐链路用 Python 循环计算，口径完整（含分辨率合并与动态范围
门限），适合小样本。本脚本把「判据侧」指标（理想 FIM 的秩、条件数、有效维数、
垂直方差份额）向量化到「按接收机真实坐标分组」的粒度，可覆盖全部约 17800 个
楼层、约 461 万条链路，用于给出跨几何的经验图谱。

分辨率受限口径（merge_by_resolution）在试点已确认对秩与 share_z 影响很小
（见 docs/notes/C2最小实验_判据失效审计.md 第 4.3 节），因此全量只跑判据侧与
动态范围受限侧，分辨率受限侧在小样本上另行复核。

输出
----
results/c2full_<dataset>_<split>.csv，逐链路的：
  n_path, los_frac, n_anchor, rx_z, mean_abs_uz, sum_w,
  lam_min, lam_mid, lam_max, d_eff, kappa, rank, share_z,
  share_x, share_y, gdop, logdet, var_z, trace_inv, floor
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"

EPS_REL = 1e-12  # 特征值相对下限，保留极小方向的惩罚


def metrics_batch(J: np.ndarray) -> dict:
    """向量化从 (G,3,3) 的 FIM 提取指标。"""
    J = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    lam, Q = np.linalg.eigh(J)          # 升序，(G,3),(G,3,3)
    lam = np.clip(lam, 0.0, None)
    lam_max = lam[:, -1]
    ok = lam_max > 0
    lam_c = np.where(ok[:, None], np.clip(lam, lam_max[:, None] * EPS_REL, None), 1.0)
    rank = (lam > (lam_max[:, None] * 1e-9)).sum(axis=1).astype(np.int64)
    kappa = np.where(ok, lam_c[:, -1] / lam_c[:, 0], np.inf)
    s2 = (lam ** 2).sum(axis=1)
    d_eff = np.where(s2 > 0, (lam.sum(axis=1) ** 2) / np.where(s2 > 0, s2, 1.0), 0.0)
    inv = 1.0 / lam_c
    # 各坐标轴的方差份额：var_a = sum_k Q[a,k]^2 / lam_k，即 e_a^T J^+ e_a
    var_x = (Q[:, 0, :] ** 2 * inv).sum(axis=1)
    var_y = (Q[:, 1, :] ** 2 * inv).sum(axis=1)
    var_z = (Q[:, 2, :] ** 2 * inv).sum(axis=1)
    trace_inv = inv.sum(axis=1)
    safe = np.where(trace_inv > 0, trace_inv, 1.0)
    share_x = var_x / safe
    share_y = var_y / safe
    share_z = np.where(trace_inv > 0, var_z / safe, 1.0)
    gdop = np.sqrt(np.where(trace_inv > 0, trace_inv, 0.0))   # sqrt(trace(J^+))
    logdet = np.log(lam_c).sum(axis=1)                        # D-optimality 代理
    return dict(rank=rank, kappa=kappa, d_eff=d_eff, share_z=share_z,
                lam_min=lam[:, 0], lam_mid=lam[:, 1], lam_max=lam[:, 2],
                var_x=var_x, var_y=var_y, var_z=var_z,
                share_x=share_x, share_y=share_y,
                gdop=gdop, logdet=logdet, trace_inv=trace_inv)


def process_floor(f: h5py.File, floor_id: int, dr_db: float):
    """返回单个楼层逐链路的指标字典（等长数组）。"""
    bounds = wio.floor_channel_bounds(f)
    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    if c1 <= c0:
        return None
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    tx_idx, rx_idx, pstart = ch[:, 0], ch[:, 1], ch[:, 2]
    M = ch.shape[0]

    utx, urx = np.unique(tx_idx), np.unique(rx_idx)
    tx_of_ch = np.searchsorted(utx, tx_idx)
    tx_xyz_ch = np.asarray(f["tx"][utx], dtype=np.float64)[tx_of_ch]
    rx_xyz = np.asarray(f["rx"][urx], dtype=np.float64)[np.searchsorted(urx, rx_idx)]

    p0 = int(pstart[0])
    if c1 < np.asarray(f["channels"].shape[0]):
        p_end = int(np.asarray(f["channels"][c1, 2]))
    else:
        p_end = int(np.asarray(f["paths"].shape[0]))
    paths = np.asarray(f["paths"][p0:p_end], dtype=np.float64)
    K = paths.shape[0]
    if K == 0:
        return None
    rel = pstart - p0
    counts = np.diff(np.append(rel, K))
    ch_of_path = np.repeat(np.arange(M, dtype=np.int64), counts)

    # 分组：按接收机真实坐标（同坐标视为同一链路组）
    _, rx_uid = np.unique(rx_xyz, axis=0, return_inverse=True)
    G = int(rx_uid.max()) + 1
    n_ch_g = np.bincount(rx_uid, minlength=G)

    # ---- 逐信道的 LoS 标记（首径时延对应距离与几何距离之差） ----
    delay_min_ch = np.minimum.reduceat(paths[:, 2], rel)
    d_geo_ch = np.linalg.norm(tx_xyz_ch - rx_xyz, axis=1)
    los_ch = (delay_min_ch * wio.C_M_PER_NS - d_geo_ch <= wio.LOS_EXCESS_M).astype(np.float64)
    los_frac_g = np.bincount(rx_uid, weights=los_ch, minlength=G) / n_ch_g
    rx_z_g = np.bincount(rx_uid, weights=rx_xyz[:, 2], minlength=G) / n_ch_g

    # ---- 逐信道的锚点数（组内不同 tx 的个数） ----
    ntx = int(utx.size)
    key = rx_uid.astype(np.int64) * ntx + tx_of_ch
    uk = np.unique(key)
    n_anchor_g = np.bincount(uk // ntx, minlength=G)

    # ---- 逐路径的组号、功率与到达方向 ----
    grp_p = rx_uid[ch_of_path]
    w = 10.0 ** (paths[:, 0] / wio.GAIN_DB_DIVISOR)
    aoa_az, zoa = paths[:, 5], paths[:, 6]
    dirs = np.stack([np.sin(zoa) * np.cos(aoa_az),
                     np.sin(zoa) * np.sin(aoa_az),
                     np.cos(zoa)], axis=1)

    order = np.argsort(grp_p, kind="stable")
    grp_s = grp_p[order]
    starts = np.flatnonzero(np.r_[True, grp_s[1:] != grp_s[:-1]])
    cnt_all = np.diff(np.append(starts, K))

    # 动态范围门限：以组内最大功率为基准
    w_s = w[order]
    g_max = np.maximum.reduceat(w_s, starts)
    thr = 10.0 ** (-dr_db / 10.0)
    keep = w_s >= (np.repeat(g_max, cnt_all) * thr)

    d_o, w_o, grp_o = dirs[order][keep], w_s[keep] / np.repeat(g_max, cnt_all)[keep], grp_s[keep]
    st2 = np.flatnonzero(np.r_[True, grp_o[1:] != grp_o[:-1]])
    cnt2 = np.diff(np.append(st2, d_o.shape[0]))
    gid = grp_o[st2]                                  # 保留下来的组号（原始编号）

    outer = (d_o[:, :, None] * d_o[:, None, :]) * w_o[:, None, None]
    J = np.add.reduceat(outer.reshape(d_o.shape[0], 9), st2, axis=0).reshape(-1, 3, 3)
    met = metrics_batch(J)

    mean_abs_uz_g = np.add.reduceat(np.abs(dirs[order][:, 2]), starts) / cnt_all
    sum_w_g = np.add.reduceat(w_s, starts)

    return dict(
        n_path=cnt2, sum_w=np.add.reduceat(w_o, st2),
        los_frac=los_frac_g[gid], n_anchor=n_anchor_g[gid],
        rx_z=rx_z_g[gid], mean_abs_uz=mean_abs_uz_g[gid],
        **{k: met[k] for k in ("rank", "kappa", "d_eff", "share_z",
                               "lam_min", "lam_mid", "lam_max",
                               "var_x", "var_y", "var_z",
                               "share_x", "share_y", "gdop", "logdet",
                               "trace_inv")},
    )


KEYS = ["n_path", "los_frac", "n_anchor", "rx_z", "mean_abs_uz", "sum_w",
        "lam_min", "lam_mid", "lam_max", "d_eff", "kappa", "rank",
        "share_z", "share_x", "share_y", "gdop", "logdet",
        "var_z", "trace_inv", "floor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="只跑指定 split；缺省跑全部")
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()

    splits = wio.discover_splits(ROOT)
    if args.split:
        splits = [s for s in splits if s.tag == args.split]
    RES.mkdir(parents=True, exist_ok=True)

    total = 0
    t0 = time.time()
    for info in splits:
        t1 = time.time()
        acc = {k: [] for k in KEYS}
        with h5py.File(info.path, "r") as f:
            flist = wio.list_floors(f)
            for fid, fidx in flist:
                r = process_floor(f, fid, args.dr)
                if r is None:
                    continue
                for k in KEYS:
                    if k == "floor":
                        acc[k].append(np.full(r["rank"].shape[0], fidx, dtype=np.int64))
                    else:
                        acc[k].append(r[k])
        if not acc["rank"]:
            print(f"[{info.tag}] 无样本")
            continue
        out = {k: np.concatenate(v) for k, v in acc.items()}
        n = out["rank"].shape[0]
        total += n
        fname = RES / f"c2full_{info.tag.replace('/', '_')}{args.tag_suffix}.csv"
        with open(fname, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(KEYS)
            for row in zip(*[out[k] for k in KEYS]):
                wr.writerow(row)
        print(f"[{info.tag}] 楼层 {len(acc['rank'])}  链路 {n}  "
              f"用时 {time.time()-t1:.1f}s  -> {fname.name}")
    print(f"\n全量完成，总链路 {total}，总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
