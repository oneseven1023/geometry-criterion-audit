#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_criterion_audit.py -- C2 最小实验：解析可辨识性判据在真实几何上的失效审计

问题
----
标准解析判据把每条多径当作一条独立、可分辨、信息量按增益加权的约束，由此得到
位置 Fisher 信息矩阵 J，并断言「J 满秩即位置局部可辨识」。真实可见性受限时：
(a) 有限带宽使时延相距 < 1/B 的路径不可分辨；
(b) 有限阵列孔径使角距 < delta_theta 的路径不可分辨；
(c) 有限动态范围使弱径不提供有效信息。
三类限制都会压低实际可达的信息矩阵，但理想判据对此不敏感。

模型
----
对每条 MPC，用时延与到达角把发射机映射为虚拟锚点：VA = rx - c*tau*u_arr，
u_arr 为由 (aoa_az, zoa) 得到的到达方向单位向量（见 notebooks/绕射与虚拟锚点
等价性算例.md 与 Aditya/Molisch Proc.IEEE 2018）。于是本链路对位置的信息为
到 VA 的测距信息：

    J_link = sum_i (1/sigma_i^2) u_i u_i^T ,  u_i = 到达方向单位向量
    sigma_i^2 ~ 1 / (B^2 * SNR_i),  SNR_i ~ 路径线性功率 w_i = 10^(gain/5)

故 J_link ∝ sum_i w_i u_i u_i^T（B 只改整体尺度，不改秩，这与单墙算例结论一致）。

判据侧（ideal）：全部路径参与，全部视为可分辨。
真实侧（effective）：先在时延与角域做分辨率合并，再施加动态范围门限。

指标（尽量做到阈值无关）
------------------------
share_z = e_z^T J^+ e_z / trace(J^+)   —— 垂直方向的位置方差份额，越大越差
kappa   = lambda_1 / lambda_3          —— 信息矩阵条件数
d_eff   = (sum lambda)^2 / sum lambda^2 —— 有效维数，1..3，无阈值

用法
----
python c2_criterion_audit.py --split WiPTR/val_grid --floors 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

C_M_PER_NS = wio.C_M_PER_NS


# ----------------------------------------------------------------------------
# 基础量
# ----------------------------------------------------------------------------
def path_directions(paths: np.ndarray) -> np.ndarray:
    """(K,7) -> (K,3) 到达方向单位向量（以 zoa 为天顶角，aoa_az 为方位角）。"""
    aoa_az = paths[:, 5]
    zoa = paths[:, 6]
    sx = np.sin(zoa) * np.cos(aoa_az)
    sy = np.sin(zoa) * np.sin(aoa_az)
    sz = np.cos(zoa)
    return np.stack([sx, sy, sz], axis=1)


def path_power(paths: np.ndarray) -> np.ndarray:
    """(K,7) -> (K,) 线性功率 w = 10^(gain/5)。"""
    return 10.0 ** (paths[:, 0] / wio.GAIN_DB_DIVISOR)


def fim(dirs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """位置 FIM = sum_i w_i u_i u_i^T。"""
    return (dirs * w[:, None]).T @ dirs


def fim_metrics(J: np.ndarray) -> dict:
    """从 3x3 FIM 提取阈值无关指标。"""
    J = 0.5 * (J + J.T)
    lam, Q = np.linalg.eigh(J)  # 升序
    lam = np.clip(lam, 0.0, None)
    lam_max = lam[-1]
    if lam_max <= 0:
        return dict(rank=0, kappa=np.inf, d_eff=0.0, share_z=1.0, lam=(0, 0, 0))
    lam_c = np.clip(lam, lam_max * 1e-12, None)  # 防除零，保留极小方向的惩罚
    rank = int(np.sum(lam > lam_max * 1e-9))
    kappa = float(lam_c[-1] / lam_c[0])
    d_eff = float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 0.0
    # J^+ 在标准基 e_z 上的二次型
    inv = 1.0 / lam_c
    var_z = float(np.sum((Q[2, :] ** 2) * inv))
    tr = float(np.sum(inv))
    share_z = var_z / tr
    return dict(rank=rank, kappa=kappa, d_eff=d_eff, share_z=share_z,
                lam=(float(lam[0]), float(lam[1]), float(lam[2])))


def merge_by_resolution(dirs, w, delay_ns, bw_hz, dtheta_deg) -> tuple[np.ndarray, np.ndarray]:
    """按 (时延 < 1/B 且 角距 < dtheta) 合并路径，返回合并后的方向与权重。

    合并组的方向取功率加权合成方向的单位向量，权重取组内功率和。
    """
    n = dirs.shape[0]
    if n == 0:
        return dirs, w
    dtau_ns = 1e9 / bw_hz  # 时延分辨率，ns
    cos_thr = np.cos(np.deg2rad(dtheta_deg))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    G = dirs @ dirs.T  # 方向余弦
    for i in range(n):
        for j in range(i + 1, n):
            if abs(delay_ns[i] - delay_ns[j]) < dtau_ns and G[i, j] > cos_thr:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_d, out_w = [], []
    for idx in groups.values():
        idx = np.asarray(idx)
        v = (dirs[idx] * w[idx, None]).sum(axis=0)
        nv = np.linalg.norm(v)
        if nv <= 0:
            continue
        out_d.append(v / nv)
        out_w.append(float(w[idx].sum()))
    return np.asarray(out_d), np.asarray(out_w)


def rx_metrics(dirs, w, delay_ns, bw_hz, dtheta_deg, dr_db) -> dict:
    """单个接收机（跨锚点全部路径）的判据侧与真实侧指标。"""
    if dirs.shape[0] == 0:
        return {}
    w = w / w.max()
    m_ideal = fim_metrics(fim(dirs, w))

    # 真实侧 1：分辨率受限（时延 + 角）
    d2, w2 = merge_by_resolution(dirs, w, delay_ns, bw_hz, dtheta_deg)
    m_res = fim_metrics(fim(d2, w2))

    # 真实侧 2：动态范围受限（丢弃弱于 dr_db 的路径）
    keep = w >= 10.0 ** (-dr_db / 10.0)
    m_dr = fim_metrics(fim(dirs[keep], w[keep])) if keep.sum() >= 1 else fim_metrics(np.zeros((3, 3)))

    # 真实侧 3：两者同时
    d3, w3 = merge_by_resolution(dirs[keep], w[keep], delay_ns[keep], bw_hz, dtheta_deg)
    m_both = fim_metrics(fim(d3, w3)) if d3.shape[0] else fim_metrics(np.zeros((3, 3)))

    return dict(
        n_path=dirs.shape[0],
        mean_abs_uz=float(np.abs(dirs[:, 2]).mean()),
        lam_ideal_min=m_ideal["lam"][0], lam_ideal_mid=m_ideal["lam"][1],
        lam_ideal_max=m_ideal["lam"][2],
        rank_ideal=m_ideal["rank"], kappa_ideal=m_ideal["kappa"],
        d_eff_ideal=m_ideal["d_eff"], share_z_ideal=m_ideal["share_z"],
        rank_res=m_res["rank"], kappa_res=m_res["kappa"],
        d_eff_res=m_res["d_eff"], share_z_res=m_res["share_z"],
        rank_dr=m_dr["rank"], share_z_dr=m_dr["share_z"],
        rank_both=m_both["rank"], d_eff_both=m_both["d_eff"], share_z_both=m_both["share_z"],
    )


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def run_split(info, floors, bw_hz, dtheta_deg, dr_db, max_rx=None):
    rows = []
    with h5py.File(info.path, "r") as f:
        flist = [fid for fid, _ in wio.list_floors(f)]
        if floors is not None:
            flist = flist[:floors]
        for fid in flist:
            fc = wio.read_floor(f, fid)
            paths, ch_of_path = wio.read_floor_paths(f, fid)
            dirs_all = path_directions(paths)
            w_all = path_power(paths)
            delay_all = paths[:, 2]
            # 按「接收机真实坐标」分组（对 val_grid 跨锚点坐标一致 -> 真多锚点；
            # 对 main_5tx 各锚点坐标不同 -> 每锚点独立，不会误并）
            _, rx_uid = np.unique(fc.rx_xyz, axis=0, return_inverse=True)
            rx_of_path = rx_uid[ch_of_path]
            los_of_path = fc.los_flag[ch_of_path]

            pids = np.unique(rx_of_path)
            if max_rx is not None:
                pids = pids[:max_rx]
            for pid in pids:
                sel = rx_of_path == pid
                d = dirs_all[sel]
                wv = w_all[sel]
                dv = delay_all[sel]
                m = rx_metrics(d, wv, dv, bw_hz, dtheta_deg, dr_db)
                if not m:
                    continue
                n_anchor = int(np.unique(fc.tx_idx[ch_of_path[sel]]).size)
                ch_sel = np.unique(ch_of_path[sel])
                rx_z = float(fc.rx_xyz[ch_sel[0], 2])
                m.update(floor_id=fid, floor_idx=fc.floor_idx, rx_pos_id=int(pid),
                         n_anchor=n_anchor, los_frac=float(los_of_path[sel].mean()),
                         rx_z=rx_z)
                rows.append(m)
    return rows


def summarize(rows: list[dict], tag: str, cfg: dict):
    import numpy as np
    n = len(rows)
    if n == 0:
        print(f"[{tag}] 无样本"); return
    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    share_i = arr("share_z_ideal"); share_r = arr("share_z_res"); share_b = arr("share_z_both")
    rank_i = arr("rank_ideal"); rank_r = arr("rank_res"); rank_b = arr("rank_both")
    kap_i = arr("kappa_ideal"); dei = arr("d_eff_ideal"); deb = arr("d_eff_both")

    print(f"\n===== [{tag}]  样本 {n}  |  B={cfg['bw_hz']/1e6:.0f} MHz "
          f"dtheta={cfg['dtheta_deg']:.0f}deg DR={cfg['dr_db']:.0f}dB =====")
    print(f"  秩(理想) 分布: " + ", ".join(
        f"r{int(r)}:{int((rank_i==r).sum())}" for r in np.unique(rank_i)))
    print(f"  秩(分辨率受限) 分布: " + ", ".join(
        f"r{int(r)}:{int((rank_r==r).sum())}" for r in np.unique(rank_r)))
    print(f"  share_z 理想  中位 {np.median(share_i):.4f}  90分位 {np.percentile(share_i,90):.4f}")
    print(f"  share_z 受限  中位 {np.median(share_r):.4f}  90分位 {np.percentile(share_r,90):.4f}")
    print(f"  share_z 双重  中位 {np.median(share_b):.4f}  90分位 {np.percentile(share_b,90):.4f}")
    print(f"  d_eff  理想  中位 {np.median(dei):.3f}  受限 中位 {np.median(deb):.3f}")
    print(f"  kappa(理想) 中位 {np.median(kap_i):.2e}")
    # 偏差关键指标
    b_rank = (rank_i == 3) & (rank_b < 3)
    b_z = (rank_i == 3) & (share_b > 0.10)
    print(f"  * 判据满秩但双重受限后秩不足 比例: {b_rank.mean():.4f}")
    print(f"  * 判据满秩但垂直份额>10% 比例: {b_z.mean():.4f}")
    # 分层：LoS / 锚点数
    los = np.array([r["los_frac"] for r in rows])
    na = np.array([r["n_anchor"] for r in rows])
    for lo, hi, name in [(0.99, 1.01, "全LoS"), (0.0, 0.01, "全NLoS")]:
        m = (los >= lo) & (los < hi)
        if m.sum() > 30:
            print(f"  [{name}] n={int(m.sum())}  share_z受限 中位 {np.median(share_r[m]):.4f}"
                  f"  d_eff受限 中位 {np.median(deb[m]):.3f}")
    for na_v in np.unique(na):
        m = na == na_v
        if m.sum() > 30:
            print(f"  [锚点{int(na_v)}] n={int(m.sum())}  share_z受限 中位 {np.median(share_r[m]):.4f}")
    # 诊断：垂直几何
    uz = arr("mean_abs_uz"); rx_z = arr("rx_z")
    print(f"  诊断 mean|u_z| 中位 {np.median(uz):.4f}   (接近 0 表示到达方向近水平)")
    print(f"  诊断 lam_ideal 中位 (min/mid/max) = "
          f"{np.median(arr('lam_ideal_min')):.2e} / {np.median(arr('lam_ideal_mid')):.2e} / "
          f"{np.median(arr('lam_ideal_max')):.2e}")
    for zv in np.unique(np.round(rx_z, 2)):
        m = np.round(rx_z, 2) == zv
        if m.sum() > 20:
            print(f"  [rx_z={zv:.2f}] n={int(m.sum())}  share_z受限 中位 {np.median(share_r[m]):.4f}"
                  f"  mean|u_z| 中位 {np.median(uz[m]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="WiPTR/val_grid")
    ap.add_argument("--floors", type=int, default=3, help="-1 表示全部楼层")
    ap.add_argument("--bw", type=float, default=100e6, help="带宽 Hz")
    ap.add_argument("--dtheta", type=float, default=20.0, help="角分辨率 度")
    ap.add_argument("--dr", type=float, default=30.0, help="动态范围 dB")
    ap.add_argument("--max-rx", type=int, default=4000, help="-1 表示全部接收机")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.floors < 0:
        args.floors = None
    if args.max_rx < 0:
        args.max_rx = None

    root = Path(__file__).resolve().parent.parent / "data" / "public_raw"
    splits = {s.tag: s for s in wio.discover_splits(root)}
    if args.split not in splits:
        print("可用 split:", list(splits)); return
    info = splits[args.split]
    cfg = dict(bw_hz=args.bw, dtheta_deg=args.dtheta, dr_db=args.dr)
    rows = run_split(info, args.floors, args.bw, args.dtheta, args.dr, args.max_rx)
    summarize(rows, info.tag, cfg)
    if args.csv:
        import csv
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader(); wr.writerows(rows)
        print(f"\n明细已写入 {args.csv}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
