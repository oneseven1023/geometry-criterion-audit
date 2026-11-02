#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_truth_capture.py -- 量化单锚点近镜像简并在噪声下的实际后果

背景
----
`c2_truth_diagnose.py` 确认：扩样中 87 条「多起点出现真值以外极小点」的链路，
其虚拟锚点 VA 点云近乎共面（s3/s1 中位 0.02），远解与「关于拟合平面的镜像点」
距离中位仅 0.055 m（98.3% 落在 0.5 m 内）。无噪声下真值残差为 0、镜像解残差
rms 约 0.046 m，二者不同量级但绝对差很小。

本脚本回答「这在实际噪声下有没有后果」：给测距加名义高斯噪声 sigma0，从真值
附近 3 m 的多个无先验初始点做高斯牛顿求解，统计收敛到真解（<0.25 m）的比例。
若真解与镜像解在噪声下不可分，则成功率会显著低于 1。

对照组：同 split、同为单锚点的非歧义链路（多起点无退化），检验成功率是否接近 1。

用法
----
python c2_truth_capture.py --csv results/c2truth_all.csv --n-init 64
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_independent_truth import gn_range_solve  # noqa: E402
from c2_truth_stratified import floor_links  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
C_M_PER_NS = wio.C_M_PER_NS


def capture_rate(L, gi, x_true, sigma0, n_init, seed, tol=0.25, max_off=3.0):
    s = L["st2"][gi]
    e = s + L["cnt2"][gi]
    dirs, w, tau = L["d_o"][s:e], L["w_o"][s:e], L["tau_o"][s:e]
    wn = w / w.max()
    VA = x_true[None, :] - C_M_PER_NS * tau[:, None] * dirs
    K = VA.shape[0]
    sig = sigma0 / np.sqrt(wn)
    dist_true = np.linalg.norm(x_true[None, :] - VA, axis=1)

    # 判据 CRB（口径同 c2_independent_truth.analyse_rx）
    J = (wn[:, None] * dirs).T @ dirs
    Jc = 0.5 * (J + J.T)
    lam, Q = np.linalg.eigh(Jc)
    Jinv = (Q * (1.0 / np.clip(lam, lam[-1] * 1e-12, None))) @ Q.T
    CRB = sigma0 ** 2 * Jinv

    rng = np.random.default_rng(seed)
    r = dist_true[None, :] + sig[None, :] * rng.normal(size=(n_init, K))
    X0 = x_true[None, :] + rng.normal(0.0, max_off, size=(n_init, 3))
    Xh = gn_range_solve(VA, r, sig, X0, iters=300, damp=1e-6)
    err = np.linalg.norm(Xh - x_true[None, :], axis=1)
    return (float((err < tol).mean()), float(np.median(err)),
            float(np.trace(CRB)), float(CRB[2, 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent.parent
                                         / "results" / "c2truth_all.csv"))
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--n-init", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max", type=int, default=0, help="每组最多取多少条；0 为全部")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    amb = [r for r in rows if int(float(r["ambig"])) == 1]
    amb.sort(key=lambda r: -int(float(r["n_other"])))
    if args.max:
        amb = amb[:args.max]
    amb_keys = {(r["split"], int(float(r["floor_id"])),
                 round(float(r["rx_x"]), 4), round(float(r["rx_y"]), 4),
                 round(float(r["rx_z"]), 4)) for r in amb}

    # 对照：同 split、单锚点、非歧义的链路，随机取与歧义组等量（按 split 配比）
    rng = np.random.default_rng(args.seed)
    ctrl = []
    by_split = defaultdict(list)
    for r in rows:
        if int(float(r["ambig"])) == 0 and int(float(r["n_anchor"])) == 1:
            by_split[r["split"]].append(r)
    want = defaultdict(int)
    for r in amb:
        want[r["split"]] += 1
    for sp, lst in by_split.items():
        k = want.get(sp, 0)
        if k == 0:
            continue
        idx = rng.choice(len(lst), size=min(k, len(lst)), replace=False)
        ctrl.extend(lst[i] for i in idx)

    splits = {s.tag: s for s in wio.discover_splits(ROOT)}

    def run_group(group, tag):
        by_floor = defaultdict(list)
        for r in group:
            by_floor[(r["split"], int(float(r["floor_id"])))].append(r)
        rates, mdists, ctr, cz = [], [], [], []
        for (sp, fid), rs in by_floor.items():
            with h5py.File(splits[sp].path, "r") as f:
                L = floor_links(f, fid, 30.0)
                if L is None:
                    continue
                for r in rs:
                    xt = np.array([float(r["rx_x"]), float(r["rx_y"]), float(r["rx_z"])])
                    d = np.linalg.norm(L["rx_xyz_g"] - xt[None, :], axis=1)
                    gi = int(np.argmin(d))
                    if d[gi] > 1e-6:
                        continue
                    cr, md, ct, czz = capture_rate(L, gi, xt, args.sigma0, args.n_init,
                                                   seed=hash((sp, fid, gi)) % (2**31))
                    rates.append(cr)
                    mdists.append(md)
                    ctr.append(ct)
                    cz.append(czz)
        rates = np.array(rates)
        mdists = np.array(mdists)
        ctr = np.array(ctr)
        cz = np.array(cz)
        print(f"\n[{tag}]  链路 {len(rates)} 条  sigma0={args.sigma0} m  "
              f"初始点 {args.n_init} 个（偏离真值 3 m 标准差）")
        print(f"  无先验条件下收敛到真解(<0.25m) 的比例：中位 {np.median(rates):.4f}  "
              f"均值 {rates.mean():.4f}")
        print(f"  成功率<0.75 的链路占比 {100*(rates<0.75).mean():.1f}%  "
              f"<0.5 的占比 {100*(rates<0.5).mean():.1f}%")
        print(f"  估计误差中位 {np.median(mdists):.3f} m   "
              f"判据 CRB sqrt(trace) 中位 {np.sqrt(np.median(ctr)):.3f} m   "
              f"CRB 垂直轴 sqrt 中位 {np.sqrt(np.median(cz)):.3f} m")
        print(f"  误差中位 / sqrt(trace CRB) 中位 {np.median(mdists/np.sqrt(ctr)):.2f}"
              f"   （≈1 表示误差与 CRB 一致，即大误差源于 CRB 大而非额外陷阱）")
        return rates

    ra = run_group(amb, "歧义组（单锚点近镜像简并）")
    rc = run_group(ctrl, "对照组（单锚点非歧义）")
    print(f"\n对比：歧义组成功率中位 {np.median(ra):.3f}  vs  对照组 {np.median(rc):.3f}")


if __name__ == "__main__":
    main()
