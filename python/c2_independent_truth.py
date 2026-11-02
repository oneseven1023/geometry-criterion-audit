#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_independent_truth.py -- 方案一：为 C2 审计补一路独立真值

背景
----
`c2_criterion_audit.py` 的真实侧是同一模型在分辨率与动态范围上的退化版本，
不是独立真值，因此只能说明判据的标量结论与实际信息结构脱节，不能说明判据
是否误导了实际可达精度。本脚本补两条不依赖 FIM 秩假设的真值。

真值一：多起点非线性求解（真正的可辨识性，模型层面）
    用虚拟锚点测距模型 r_i = |x - VA_i|，从真值附近的多个初始点做高斯牛顿
    求解。若从多个不同初始点都收敛回真值，则真值唯一可辨识；若存在收敛到
    真值以外的极小点，则该接收机在模型层面确实不可辨识。这与 FIM 的秩无关。

真值二：蒙特卡洛可达精度（判据 CRB 是否脱节）
    给测距加高斯噪声，重复求解得到位置估计的样本协方差，与判据 CRB 比较。
    比值显著大于一，说明判据 CRB 对该几何过乐观；比值接近一，说明判据 CRB
    本身是准的，问题只出在把它读成秩或条件数这类标量。

旁证三：数据驱动的垂直敏感度
    直接在数据上把每锚点首径时延对位置回归，看垂直方向的一阶敏感度相对水平
    方向的比值，作为不依赖任何模型的第三路旁证。

用法
----
python c2_independent_truth.py --split WiPTR/val_grid --floors 2 --max-rx 300
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_criterion_audit import path_directions, path_power  # noqa: E402

C_M_PER_NS = wio.C_M_PER_NS


def gn_range_solve(VA: np.ndarray, r: np.ndarray, sig: np.ndarray,
                   X0: np.ndarray, iters: int = 80, damp: float = 1e-4) -> np.ndarray:
    """向量化高斯牛顿（含莱文贝格阻尼）：未知 X (T,3)，锚点 VA (K,3)。

    最小化 sum_k ((|X - VA_k| - r_k) / sig_k)^2。
    """
    X = X0.copy()
    eye = np.eye(3)[None]
    for _ in range(iters):
        D = X[:, None, :] - VA[None, :, :]          # (T,K,3)
        dist = np.linalg.norm(D, axis=2)            # (T,K)
        dist = np.maximum(dist, 1e-9)
        res = (dist - r) / sig[None, :]             # (T,K)
        Jm = D / dist[:, :, None] / sig[None, :, None]    # (T,K,3) = d(res)/dX
        H = np.einsum("tki,tkj->tij", Jm, Jm)
        g = np.einsum("tki,tk->ti", Jm, res)
        # 莱文贝格阻尼：按 H 的迹取相对量，保证近奇异方向不炸
        tr = np.trace(H, axis1=1, axis2=2) / 3.0
        Hd = H + (damp * tr)[:, None, None] * eye
        dx = np.linalg.solve(Hd, -g[..., None])[..., 0]
        X = X + dx
    return X


def rx_setup(dirs, w, delay, x_true):
    """由路径构造虚拟锚点与权重，返回 VA, sig 的相对形状。"""
    VA = x_true[None, :] - C_M_PER_NS * delay[:, None] * dirs
    wn = w / w.max()
    return VA, wn


def analyse_rx(dirs, w, delay, x_true, sigma0, n_mc=200, n_init=64, seed=0):
    rng = np.random.default_rng(seed)
    VA, wn = rx_setup(dirs, w, delay, x_true)
    K = VA.shape[0]
    sig = sigma0 / np.sqrt(wn)                      # (K,)
    dist_true = np.linalg.norm(x_true[None, :] - VA, axis=1)

    # --- 判据 CRB ---
    J = (wn[:, None] * dirs).T @ dirs               # 3x3
    Jc = 0.5 * (J + J.T)
    lam, Q = np.linalg.eigh(Jc)
    lam_max = lam[-1]
    lam_c = np.clip(lam, lam_max * 1e-12, None)
    Jinv = (Q * (1.0 / lam_c)) @ Q.T
    CRB = sigma0 ** 2 * Jinv
    crb_z = float(CRB[2, 2])
    rank = int(np.sum(lam > lam_max * 1e-9))

    # --- 真值一：多起点无噪声求解，检测真值以外的极小点 ---
    off = rng.normal(0.0, 3.0, size=(n_init, 3))
    X0 = x_true[None, :] + off
    r_b = np.broadcast_to(dist_true, (n_init, K))
    Xh = gn_range_solve(VA, r_b, np.ones(K), X0, iters=300, damp=1e-6)
    err = np.linalg.norm(Xh - x_true[None, :], axis=1)
    n_other = int(np.sum(err > 0.25))               # 收敛到真值以外的初始点个数
    ambig = n_other > 0
    # 若存在其他解，记录其与真值的距离与垂直偏移
    other_dz = float(np.median(Xh[err > 0.25, 2] - x_true[2])) if n_other else 0.0

    # --- 真值二：蒙特卡洛可达精度（每个试验独立初始点，避免初始点造成的偏置） ---
    r_mc = dist_true[None, :] + sig[None, :] * rng.normal(size=(n_mc, K))
    X0m = x_true[None, :] + rng.normal(0, 0.02, size=(n_mc, 3))
    Xm = gn_range_solve(VA, r_mc, sig, X0m, iters=500, damp=1e-7)
    dX = Xm - x_true[None, :]
    fail = np.linalg.norm(dX, axis=1) > 5.0
    ok = ~fail
    var = np.var(dX[ok], axis=0) if ok.sum() > 10 else np.full(3, np.inf)
    ratio = var / np.array([CRB[0, 0], CRB[1, 1], crb_z])
    bias = np.mean(dX[ok], axis=0) if ok.sum() > 10 else np.full(3, np.nan)
    maha = float(np.mean(np.einsum("ti,ij,tj->t", dX[ok], Jinv, dX[ok]) / 3.0)) \
        if ok.sum() > 10 else float("inf")

    return dict(
        n_path=K, rank=rank, crb_z=crb_z,
        share_z=float(crb_z / np.trace(CRB)),
        ambig=int(ambig), n_other=n_other, other_dz=other_dz,
        mc_fail_frac=float(fail.mean()),
        ratio_x=float(ratio[0]), ratio_y=float(ratio[1]), ratio_z=float(ratio[2]),
        bias_norm=float(np.linalg.norm(bias)), maha=maha,
    )


def data_driven_z_sensitivity(info, floor_id):
    """数据驱动旁证：每锚点首径时延对位置回归，比较垂直与水平的一阶敏感度。"""
    with h5py.File(info.path, "r") as f:
        fc = wio.read_floor(f, floor_id)
    out = []
    for tx in np.unique(fc.tx_idx):
        m = fc.tx_idx == tx
        P = fc.rx_xyz[m]                                  # (n,3)
        y = fc.tof_first_ns[m]
        A = np.column_stack([np.ones(m.sum()), P])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        # 标准化敏感度：系数乘上该方向的位置跨度
        span = P.std(axis=0) + 1e-9
        sens = coef[1:] * span
        hor = np.hypot(sens[0], sens[1]) + 1e-12
        out.append(dict(floor=floor_id, tx=int(tx), n=int(m.sum()),
                        s_x=abs(sens[0]), s_y=abs(sens[1]), s_z=abs(sens[2]),
                        ratio_z_hor=sens[2] / hor))
    return out


def run_split(info, floors, max_rx, sigma0, n_mc, n_init, limit_min=6):
    rows = []
    with h5py.File(info.path, "r") as f:
        flist = [fid for fid, _ in wio.list_floors(f)]
        for fid in flist[:floors]:
            fc = wio.read_floor(f, fid)
            paths, ch_of_path = wio.read_floor_paths(f, fid)
            dirs_all = path_directions(paths)
            w_all = path_power(paths)
            delay_all = paths[:, 2]
            _, rxu = np.unique(fc.rx_xyz, axis=0, return_inverse=True)
            rx_of_path = rxu[ch_of_path]
            pids = np.unique(rx_of_path)
            if max_rx:
                pids = pids[:max_rx]
            for pid in pids:
                sel = rx_of_path == pid
                if sel.sum() < limit_min:
                    continue
                ch = np.unique(ch_of_path[sel])
                x_true = fc.rx_xyz[ch[0]].astype(float)
                r = analyse_rx(dirs_all[sel], w_all[sel], delay_all[sel], x_true,
                               sigma0, n_mc=n_mc, n_init=n_init, seed=int(pid))
                r.update(floor_id=fid, rx_x=x_true[0], rx_y=x_true[1], rx_z=x_true[2],
                         n_anchor=int(np.unique(fc.tx_idx[ch]).size))
                rows.append(r)
    return rows


def summarize(rows, tag, cfg):
    n = len(rows)
    if n == 0:
        print("无样本"); return
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    sz = col("share_z"); amb = col("ambig"); mf = col("mc_fail_frac")
    rz = col("ratio_z"); rx_ = col("ratio_x"); ry = col("ratio_y"); ma = col("maha")
    rk = col("rank")
    fin = np.isfinite(rz)
    print(f"\n===== [{tag}]  n={n}  sigma0={cfg['sigma0']} m  "
          f"(多起点 {cfg['n_init']} 次, 蒙特卡洛 {cfg['n_mc']} 次) =====")
    print(f"  秩分布: " + ", ".join(f"r{int(v)}:{int((rk==v).sum())}" for v in np.unique(rk)))
    print(f"  多起点发现真值以外解 的比例: {amb.mean():.4f}   （{int(amb.sum())}/{n}）")
    print(f"  蒙特卡洛不收敛(离真值>5m) 比例: 中位 {np.median(mf):.4f}  最大 {mf.max():.4f}")
    print(f"  Mahalanobis/3 中位 {np.median(ma[fin]):.3f}  (理论值 1.0，越大说明 CRB 越乐观)")
    print(f"  估计偏置范数 中位 {np.median(col('bias_norm')[fin]):.4f} m  (应接近 0)")
    print(f"  ratio = 实测方差 / CRB 中位:  x {np.median(rx_[fin]):.3f}  "
          f"y {np.median(ry[fin]):.3f}  z {np.median(rz[fin]):.3f}")
    # 按垂直份额分层
    for lo, hi, name in [(0.0, 0.5, "share_z<0.5"), (0.5, 0.8, "0.5<=share_z<0.8"),
                         (0.8, 1.01, "share_z>=0.8")]:
        m = (sz >= lo) & (sz < hi) & fin
        if m.sum() > 20:
            print(f"  [{name}] n={int(m.sum())}  多起点歧义 {amb[m].mean():.3f}  "
                  f"ratio_z 中位 {np.median(rz[m]):.3f}  Maha/3 中位 {np.median(ma[m]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="WiPTR/val_grid")
    ap.add_argument("--floors", type=int, default=2)
    ap.add_argument("--max-rx", type=int, default=300)
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--n-init", type=int, default=64)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dd", action="store_true", help="额外做数据驱动垂直敏感度")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent / "data" / "public_raw"
    splits = {s.tag: s for s in wio.discover_splits(root)}
    info = splits[args.split]
    cfg = dict(sigma0=args.sigma0, n_mc=args.n_mc, n_init=args.n_init)
    rows = run_split(info, args.floors, args.max_rx, args.sigma0, args.n_mc, args.n_init)
    summarize(rows, info.tag, cfg)
    if args.csv:
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
        print(f"明细写入 {args.csv}")
    if args.dd:
        print("\n--- 数据驱动旁证：每锚点首径时延对位置的标准化敏感度 ---")
        allr = []
        with h5py.File(info.path, "r") as f:
            flist = [fid for fid, _ in wio.list_floors(f)][:args.floors]
        for fid in flist:
            allr.extend(data_driven_z_sensitivity(info, fid))
        r = np.array([x["ratio_z_hor"] for x in allr])
        print(f"  |s_z|/|s_hor| 中位 {np.median(np.abs(r)):.4f}  "
              f"（远小于 1 说明数据本身垂直一阶敏感度极弱）")
        print(f"  s_z 绝对值中位 {np.median([abs(x['s_z']) for x in allr]):.4f} ns/m,  "
              f"s_x/s_y 中位 {np.median([np.hypot(x['s_x'],x['s_y']) for x in allr]):.4f} ns/m")


if __name__ == "__main__":
    main()
