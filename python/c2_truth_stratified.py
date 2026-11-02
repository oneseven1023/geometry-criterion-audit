#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_truth_stratified.py -- 支柱二扩样：分层大样本独立真值检验

背景
----
`c2_independent_truth.py` 已在试点（val_grid 240 + val_ood 80 + main_5tx 80 = 400 条）
上给出三路独立真值的一致结论：判据 CRB 准确（逐轴方差比约为一）、真实位置在模型
层面唯一可辨识（多起点未见真值以外极小点）、垂直一阶敏感度弱是数据性质。

相对 461 万条链路的全量图谱，400 条样本太薄。本脚本把真值检验扩到分层大样本：

分层口径
--------
与全量图谱 `c2_fullscale_vec.py` 严格一致（同一向量化指标、同一动态范围门限
dr=30 dB、同一 LoS 判据），保证抽样链路的 share_z / d_eff / kappa / rank 与
图谱中的定义可直接对照。

每个 split 均匀抽 F 个楼层；每楼层内按 share_z 三档（<0.5 / 0.5~0.8 / >=0.8）
各随机抽 q 条链路，强制覆盖垂直退化端。对选中链路跑三路真值。

用法
----
python c2_truth_stratified.py --floors-per-split 40 --per-bin 6 --workers 8
python c2_truth_stratified.py --splits WiPTR/val_ood --floors-per-split 4 --per-bin 3
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_independent_truth import analyse_rx  # noqa: E402
from c2_fullscale_vec import metrics_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"

BINS = [(0.0, 0.5, "low"), (0.5, 0.8, "mid"), (0.8, 1.0001, "high")]

_FILES: dict = {}


def _get_file(path: str) -> h5py.File:
    f = _FILES.get(path)
    if f is None:
        f = h5py.File(path, "r")
        _FILES[path] = f
    return f


def floor_links(f: h5py.File, floor_id: int, dr_db: float):
    """单个楼层的逐链路 cheap 指标 + 供真值用的路径切片（口径同全量图谱）。"""
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

    _, rx_uid = np.unique(rx_xyz, axis=0, return_inverse=True)
    G = int(rx_uid.max()) + 1
    n_ch_g = np.bincount(rx_uid, minlength=G)

    delay_min_ch = np.minimum.reduceat(paths[:, 2], rel)
    d_geo_ch = np.linalg.norm(tx_xyz_ch - rx_xyz, axis=1)
    los_ch = (delay_min_ch * wio.C_M_PER_NS - d_geo_ch <= wio.LOS_EXCESS_M).astype(np.float64)
    los_frac_g = np.bincount(rx_uid, weights=los_ch, minlength=G) / n_ch_g

    rx_xyz_g = np.zeros((G, 3))
    rx_xyz_g[rx_uid] = rx_xyz

    ntx = int(utx.size)
    key = rx_uid.astype(np.int64) * ntx + tx_of_ch
    uk = np.unique(key)
    n_anchor_g = np.bincount(uk // ntx, minlength=G)

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

    w_s = w[order]
    g_max = np.maximum.reduceat(w_s, starts)
    thr = 10.0 ** (-dr_db / 10.0)
    keep = w_s >= (np.repeat(g_max, cnt_all) * thr)

    d_o = dirs[order][keep]
    w_o = w_s[keep] / np.repeat(g_max, cnt_all)[keep]
    tau_o = paths[order][keep][:, 2]
    grp_o = grp_s[keep]
    st2 = np.flatnonzero(np.r_[True, grp_o[1:] != grp_o[:-1]])
    cnt2 = np.diff(np.append(st2, d_o.shape[0]))
    gid = grp_o[st2]

    outer = (d_o[:, :, None] * d_o[:, None, :]) * w_o[:, None, None]
    J = np.add.reduceat(outer.reshape(d_o.shape[0], 9), st2, axis=0).reshape(-1, 3, 3)
    met = metrics_batch(J)

    return dict(
        n_path=cnt2, rx_xyz_g=rx_xyz_g[gid], n_anchor=n_anchor_g[gid],
        los_frac=los_frac_g[gid], d_o=d_o, w_o=w_o, tau_o=tau_o, st2=st2, cnt2=cnt2,
        **{k: met[k] for k in ("rank", "kappa", "d_eff", "share_z")},
    )


def work(task):
    (tag, path, floor_id, dr_db, per_bin, sigma0, n_mc, n_init, limit_min) = task
    f = _get_file(path)
    L = floor_links(f, floor_id, dr_db)
    if L is None:
        return []
    sz = L["share_z"]
    rng = np.random.default_rng(floor_id * 7919 + 13)
    rows = []
    for lo, hi, nm in BINS:
        idx = np.flatnonzero((sz >= lo) & (sz < hi))
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(per_bin, idx.size), replace=False)
        for gi in take:
            if L["n_path"][gi] < limit_min:
                continue
            s = L["st2"][gi]
            e = s + L["cnt2"][gi]
            dirs = L["d_o"][s:e]
            w = L["w_o"][s:e]
            tau = L["tau_o"][s:e]
            x_true = L["rx_xyz_g"][gi]
            r = analyse_rx(dirs, w, tau, x_true, sigma0,
                           n_mc=n_mc, n_init=n_init, seed=int(gi) * 97 + 1)
            r.update(split=tag, floor_id=int(floor_id),
                     rx_x=float(x_true[0]), rx_y=float(x_true[1]), rx_z=float(x_true[2]),
                     n_anchor=int(L["n_anchor"][gi]), los_frac=float(L["los_frac"][gi]),
                     share_z_c=float(sz[gi]), d_eff_c=float(L["d_eff"][gi]),
                     kappa_c=float(L["kappa"][gi]), rank_c=int(L["rank"][gi]), share_bin=nm)
            rows.append(r)
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        print("无样本"); return
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    sz = col("share_z_c"); amb = col("ambig")
    rx_ = col("ratio_x"); ry = col("ratio_y"); rz = col("ratio_z")
    ma = col("maha"); rk = col("rank_c"); bn = col("bias_norm")
    fin = np.isfinite(rz) & np.isfinite(rx_) & np.isfinite(ry)

    print(f"\n===== 全局 n={n} =====")
    print(f"  秩分布: " + ", ".join(f"r{int(v)}:{int((rk==v).sum())}" for v in np.unique(rk)))
    print(f"  多起点发现真值以外极小点 比例 {amb.mean():.4f}  ({int(amb.sum())}/{n})")
    print(f"  逐轴方差比 中位:  x {np.median(rx_[fin]):.3f}  y {np.median(ry[fin]):.3f}  "
          f"z {np.median(rz[fin]):.3f}   (1.0 = CRB 准确)")
    print(f"  逐轴方差比 90 分位:  z {np.percentile(rz[fin],90):.3f}")
    print(f"  估计偏置范数 中位 {np.median(bn[fin]):.5f} m")
    print(f"  Mahalanobis/3 中位 {np.median(ma[fin]):.3f}  (重尾，仅作旁证)")

    print(f"\n  按 split 分层:")
    print(f"  {'split':<28}{'n':>6}{'歧义%':>8}{'ratio_z':>9}{'ratio_xy':>9}"
          f"{'share_z':>9}{'d_eff':>7}")
    for tg in sorted({r["split"] for r in rows}):
        m = np.array([r["split"] == tg for r in rows])
        if m.sum() < 5:
            continue
        rzm = np.array([rows[i]["ratio_z"] for i in np.flatnonzero(m)])
        rxy = np.array([np.hypot(rows[i]["ratio_x"], rows[i]["ratio_y"])
                        for i in np.flatnonzero(m)])
        f2 = np.isfinite(rzm) & np.isfinite(rxy)
        print(f"  {tg:<28}{int(m.sum()):>6}{100*amb[m].mean():>8.2f}"
              f"{np.median(rzm[f2]):>9.3f}{np.median(rxy[f2]):>9.3f}"
              f"{np.median(sz[m]):>9.4f}{np.median(col('d_eff_c')[m]):>7.3f}")

    print(f"\n  按 share_z 档分层（关键：退化端是否出现歧义或 CRB 偏移）:")
    for nm in ("low", "mid", "high"):
        m = np.array([r["share_bin"] == nm for r in rows])
        if m.sum() < 5:
            continue
        rzm = np.array([rows[i]["ratio_z"] for i in np.flatnonzero(m)])
        f2 = np.isfinite(rzm)
        print(f"  [{nm:<4}] n={int(m.sum()):>5}  歧义 {100*amb[m].mean():.2f}%  "
              f"ratio_z 中位 {np.median(rzm[f2]):.3f}  90分位 {np.percentile(rzm[f2],90):.3f}  "
              f"share_z 中位 {np.median(sz[m]):.4f}")

    print(f"\n  按视距分层:")
    los = np.array([r["los_frac"] for r in rows])
    for lo, hi, nm in [(0.99, 1.01, "全LoS"), (0.0, 0.01, "全NLoS"), (0.01, 0.99, "混合")]:
        m = (los >= lo) & (los < hi)
        if m.sum() < 5:
            continue
        rzm = np.array([rows[i]["ratio_z"] for i in np.flatnonzero(m)])
        f2 = np.isfinite(rzm)
        print(f"  [{nm:<6}] n={int(m.sum()):>5}  歧义 {100*amb[m].mean():.2f}%  "
              f"ratio_z 中位 {np.median(rzm[f2]):.3f}  share_z 中位 {np.median(sz[m]):.4f}")

    print(f"\n  按锚点数分层:")
    na = np.array([r["n_anchor"] for r in rows])
    for v in np.unique(na):
        m = na == v
        if m.sum() < 5:
            continue
        print(f"  [锚点{int(v):<2}] n={int(m.sum()):>5}  歧义 {100*amb[m].mean():.2f}%  "
              f"ratio_z 中位 {np.median(rz[m & fin]):.3f}  share_z 中位 {np.median(sz[m]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=None, help="逗号分隔；缺省全部")
    ap.add_argument("--floors-per-split", type=int, default=40)
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--n-init", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-min", type=int, default=6)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    splits = wio.discover_splits(ROOT)
    if args.splits:
        want = set(args.splits.split(","))
        splits = [s for s in splits if s.tag in want]

    tasks = []
    for info in splits:
        with h5py.File(info.path, "r") as f:
            nf = len(wio.list_floors(f))
        F = min(args.floors_per_split, nf)
        pick = np.unique(np.linspace(0, nf - 1, F).astype(int))
        for fid in pick:
            tasks.append((info.tag, str(info.path), int(fid), args.dr,
                          args.per_bin, args.sigma0, args.n_mc, args.n_init, args.limit_min))
    print(f"split 数 {len(splits)}  任务（楼层）数 {len(tasks)}  "
          f"每层每档最多 {args.per_bin} 条  最大可能样本 "
          f"{len(tasks)*args.per_bin*3} 条")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 50 == 0:
                print(f"  ... 完成 {done}/{len(tasks)} 楼层  累计样本 {len(rows)}  "
                      f"用时 {time.time()-t0:.0f}s")
    print(f"抽样完成，样本 {len(rows)} 条，用时 {time.time()-t0:.0f}s")

    summarize(rows)
    if args.csv and rows:
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n明细写入 {args.csv}")


if __name__ == "__main__":
    main()
