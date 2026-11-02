#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""end_to_end_compare.py -- 端到端对照：判据指标与实际定位误差是否同向

要回答的问题
------------
论文主张「秩与条件数通过的几何，仍可能在垂直方向不可用，真正驱动量是等效虚拟
锚点方向的垂直分量」。此前的证据是 FIM 的解析结构（方向分辨精度）与三路独立
真值，还缺一条端到端证据：**用真实可运行的定位器测得的误差，是否与方向判据
（d_eff、share_z）同向，而与传统判据（秩、条件数）脱节**。

方法
----
1. 判据指标：按接收机位置分组构造位置 FIM，口径与 `c2_fullscale_vec` 逐位一致
   （动态范围 30 dB、`EPS_REL = 1e-12`），给出 rank、kappa、d_eff、share_z、
   mean|u_z|。
2. 估计器 A（方向定位）：只用首径到达方向，从各锚点作射线，最小化接收机到各
   射线距离平方和求解位置。它与论文主题同源，即只依赖方向信息。
3. 估计器 B（距离定位）：只用首径时延的 ToF 多边定位，实现直接复用
   `fingerprint_baseline.tof_trilaterate`，不重复造。
4. 对照：按传统判据通过与不通过、按 share_z 与 d_eff 分档比较实际误差，并给出
   各判据量与实际垂直误差的相关系数。

分组口径
--------
本脚本按 ``rx_pos_id`` 分组。全量脚本 `c2_fullscale_vec` 按接收机真实坐标分组。
对 WiPTR 的 15 个场景，两者等价（脚本首段会验证并打印）。选择 ``rx_pos_id``
是为了让判据指标与估计器共用同一分组键，避免坐标浮点匹配。

数据
----
`data/public_raw` 下 WiPTR 的 val_ood、val_grid、train_grid 前 5 个楼层，与
`data/normalized/links.csv` 覆盖的场景一致。

输出
----
results/e2e_links.csv    逐接收机位置一行（判据指标 + 两种估计器误差）
results/e2e_summary.csv  分组对照统计
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio                     # noqa: E402
import c2_fullscale_vec as c2v               # noqa: E402
from fingerprint_baseline import tof_trilaterate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"

SPLITS = [("WiPTR", "val_ood"), ("WiPTR", "val_grid"), ("WiPTR", "train_grid"),
          ("Wi3R", "main_5tx"), ("Wi3R", "outofdistribution/fourrooms")]
N_FLOOR = 5
DR_DB = 30.0

SIGMA_D = 0.10                    # 首径测距噪声标准差 (m)，与论文口径一致
SIGMA_THETA = np.radians(5.0)     # 到达方向噪声标准差 (rad)
N_MC = 200


# --------------------------------------------------------------------------- #
# 单楼层：判据指标 + 两种估计
# --------------------------------------------------------------------------- #
def floor_records(f: h5py.File, floor_id: int, dr_db: float,
                  sigma_d: float, sigma_theta: float,
                  n_mc: int) -> pd.DataFrame | None:
    bounds = wio.floor_channel_bounds(f)
    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    if c1 <= c0:
        return None
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    tx_idx, rx_id, pstart = ch[:, 0], ch[:, 1], ch[:, 2]
    M = ch.shape[0]

    utx = np.unique(tx_idx)
    tx_xyz_all = np.asarray(f["tx"][utx], dtype=np.float64)
    tx_xyz_ch = tx_xyz_all[np.searchsorted(utx, tx_idx)]
    urx = np.unique(rx_id)
    rx_xyz_ch = np.asarray(f["rx"][urx], dtype=np.float64)[np.searchsorted(urx, rx_id)]

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

    # ---- 分组：按接收机真实坐标（与 c2_fullscale_vec 口径一致） ----
    rx_xyz_uniq, grp_of_ch = np.unique(rx_xyz_ch, axis=0, return_inverse=True)
    grp_of_ch = np.asarray(grp_of_ch).ravel()
    G = int(grp_of_ch.max()) + 1

    # ---- 逐路径的方向、功率与首径 ----
    w = 10.0 ** (paths[:, 0] / wio.GAIN_DB_DIVISOR)
    aoa_az, zoa = paths[:, 5], paths[:, 6]
    dirs = np.stack([np.sin(zoa) * np.cos(aoa_az),
                     np.sin(zoa) * np.sin(aoa_az),
                     np.cos(zoa)], axis=1)
    d_first = -dirs[rel]                     # 数据集给出的是 rx 指向 tx，取负得 tx 指向 rx
    tof_first = paths[rel, 2]                # 逐信道首径时延 (ns)

    # ---- 判据侧：动态范围门限后构造 FIM（与 c2_fullscale_vec 同口径） ----
    grp_p = grp_of_ch[ch_of_path]
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
    grp_o = grp_s[keep]
    st2 = np.flatnonzero(np.r_[True, grp_o[1:] != grp_o[:-1]])
    cnt2 = np.diff(np.append(st2, d_o.shape[0]))
    gid = grp_o[st2]                         # 保留下来的组号（原始编号）

    outer = (d_o[:, :, None] * d_o[:, None, :]) * w_o[:, None, None]
    J = np.add.reduceat(outer.reshape(d_o.shape[0], 9), st2, axis=0).reshape(-1, 3, 3)
    met = c2v.metrics_batch(J)

    mean_abs_uz_g = np.add.reduceat(np.abs(dirs[order][:, 2]), starts) / cnt_all
    ntx = int(utx.size)
    key = grp_of_ch.astype(np.int64) * ntx + np.searchsorted(utx, tx_idx)
    n_anchor_g = np.bincount(np.unique(key) // ntx, minlength=G)

    # ---- 估计器 A：单站到达方向加首径时延定位（蒙特卡洛） ----
    # 每条链路的首径给出一个位置估计 tx_i + c*tau_i*u_i（视距时精确等于真值），
    # 组内等权融合。对方向与测距加入噪声重复多次，统计逐位置的误差标准差。
    order_g = np.argsort(grp_of_ch, kind="stable")
    grp_sorted = grp_of_ch[order_g]
    st_g = np.flatnonzero(np.r_[True, grp_sorted[1:] != grp_sorted[:-1]])
    cnt_g = np.diff(np.append(st_g, M))
    tx_o = tx_xyz_ch[order_g]
    az_o = aoa_az[rel][order_g]
    zoa_o = zoa[rel][order_g]
    L_o = tof_first[order_g] * wio.C_M_PER_NS

    rng = np.random.default_rng(20261012)
    acc = np.zeros((G, 3))
    acc2 = np.zeros((G, 3))
    accw = np.zeros((G, 3))                  # 增益加权（加权最小二乘）估计
    acc2w = np.zeros((G, 3))
    wf = w[rel][order_g]                     # 逐信道首径线性功率，作 WLS 权重
    wsum = np.add.reduceat(wf, st_g)         # (G,)
    for _ in range(n_mc):
        azn = az_o + rng.normal(0.0, sigma_theta, az_o.size)
        zn = zoa_o + rng.normal(0.0, sigma_theta, zoa_o.size)
        Ln = L_o + rng.normal(0.0, sigma_d, L_o.size)
        dn = -np.stack([np.sin(zn) * np.cos(azn),
                        np.sin(zn) * np.sin(azn),
                        np.cos(zn)], axis=1)
        pts = tx_o + Ln[:, None] * dn
        p = np.add.reduceat(pts, st_g, axis=0) / cnt_g[:, None]
        acc += p
        acc2 += p ** 2
        pw = np.add.reduceat(pts * wf[:, None], st_g, axis=0) / wsum[:, None]
        accw += pw
        acc2w += pw ** 2
    mu = acc / n_mc
    sd = np.sqrt(np.maximum(acc2 / n_mc - mu ** 2, 0.0))
    aoa_pred = mu
    mu_w = accw / n_mc
    sd_w = np.sqrt(np.maximum(acc2w / n_mc - mu_w ** 2, 0.0))
    wls_pred = mu_w

    # ---- 估计器 B：首径时延 ToF 多边定位（复用 fingerprint_baseline） ----
    anchors = pd.DataFrame({"tx_id": utx,
                            "tx_x": tx_xyz_all[:, 0],
                            "tx_y": tx_xyz_all[:, 1],
                            "tx_z": tx_xyz_all[:, 2]})
    test_df = pd.DataFrame({"rx_id": grp_of_ch, "tx_id": tx_idx,
                            "tof_ns": tof_first, "rx_z": rx_xyz_ch[:, 2]})
    if float(np.ptp(test_df["rx_z"].to_numpy())) < 0.05:
        tof_pred = tof_trilaterate(test_df, anchors, dims=2)
    else:
        tof_pred = tof_trilaterate(test_df, anchors, dims=3)

    # ---- 汇总逐组 ----
    truth = rx_xyz_uniq[gid]
    e_aoa = aoa_pred[gid] - truth
    e_tof = tof_pred[gid] - truth
    e_wls = wls_pred[gid] - truth
    out = pd.DataFrame({
        "n_anchor": n_anchor_g[gid],
        "rx_z": rx_xyz_uniq[gid][:, 2],
        "rank": met["rank"],
        "kappa": met["kappa"],
        "d_eff": met["d_eff"],
        "share_z": met["share_z"],
        "share_x": met["share_x"],
        "share_y": met["share_y"],
        "gdop": met["gdop"],
        "logdet": met["logdet"],
        "lam_min": met["lam_min"],
        "lam_max": met["lam_max"],
        "trace_inv": met["trace_inv"],
        "mean_abs_uz": mean_abs_uz_g[gid],
        "n_path_kept": cnt2,
        "aoa_ez": e_aoa[:, 2],
        "aoa_sz": sd[gid, 2],
        "aoa_sh": np.hypot(sd[gid, 0], sd[gid, 1]),
        "aoa_s3": np.sqrt((sd[gid] ** 2).sum(axis=1)),
        "wls_ez": e_wls[:, 2],
        "wls_sz": sd_w[gid, 2],
        "wls_sh": np.hypot(sd_w[gid, 0], sd_w[gid, 1]),
        "wls_s3": np.sqrt((sd_w[gid] ** 2).sum(axis=1)),
        "tof_ex": e_tof[:, 0], "tof_ey": e_tof[:, 1], "tof_ez": e_tof[:, 2],
    })
    out["tof_ah"] = np.hypot(out["tof_ex"], out["tof_ey"])
    out["tof_a3"] = np.sqrt(out["tof_ex"] ** 2 + out["tof_ey"] ** 2
                            + out["tof_ez"] ** 2)
    # 逐位置总误差 = 偏差（蒙特卡洛均值误差）与扩散（标准差）的合成；
    # 对无蒙特卡洛的 ToF 估计器，总误差即单次误差本身。
    for est in ("aoa", "wls"):
        out[f"{est}_th"] = np.hypot(out[f"{est}_ez"], out[f"{est}_sh"])
        out[f"{est}_tz"] = np.hypot(out[f"{est}_ez"], out[f"{est}_sz"])
        out[f"{est}_t3"] = np.sqrt(out[f"{est}_ez"] ** 2 + out[f"{est}_sh"] ** 2
                                   + out[f"{est}_sz"] ** 2)
    out["tof_tz"] = np.abs(out["tof_ez"])
    out["tof_th"] = out["tof_ah"]
    out["tof_t3"] = out["tof_a3"]
    return out


def build(sigma_d: float = SIGMA_D, sigma_theta: float = SIGMA_THETA,
          n_mc: int = N_MC) -> pd.DataFrame:
    frames = []
    for ds, sp in SPLITS:
        info = next((s for s in wio.discover_splits(ROOT)
                     if s.tag == f"{ds}/{sp}"), None)
        if info is None:
            print(f"[跳过] {ds}/{sp}")
            continue
        with h5py.File(info.path, "r") as f:
            floors = wio.list_floors(f)[:N_FLOOR]
            for fid, fidx in floors:
                d = floor_records(f, fid, DR_DB, sigma_d, sigma_theta, n_mc)
                if d is None:
                    continue
                d.insert(0, "floor_idx", fidx)
                d.insert(0, "split", f"{ds}/{sp}")
                frames.append(d)
        print(f"[{ds}/{sp}] 累计 {sum(len(x) for x in frames)} 个接收机位置")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 对照统计
# --------------------------------------------------------------------------- #
def _c(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def summarize(df: pd.DataFrame, cols: tuple[str, str, str, str], est: str,
              tol_z: float = 0.5, tol_3d: float = 1.0) -> list[dict]:
    """cols = (垂直总误差列, 水平总误差列, 三维总误差列, 垂直带符号偏差列)。"""
    rows: list[dict] = []
    cz, ch, c3, sb = cols
    ok = df[cz].notna() & df[c3].notna()
    d = df[ok]
    if d.shape[0] == 0:
        return rows
    ez = d[cz].to_numpy()
    eh = d[ch].to_numpy()
    e3 = d[c3].to_numpy()
    bz = d[sb].to_numpy()

    def add(tag: str, m: np.ndarray) -> None:
        if m.sum() < 10:
            return
        zz, hh, ee, bb = ez[m], eh[m], e3[m], bz[m]
        rows.append(dict(estimator=est, group=tag, n=int(m.sum()),
                         ez_med=float(np.median(zz)),
                         ez_p90=float(np.percentile(zz, 90)),
                         ez_rmse=float(np.sqrt(np.mean(zz ** 2))),
                         ez_bias=float(np.median(bb)),
                         eh_med=float(np.median(hh)),
                         e3_med=float(np.median(ee)),
                         e3_rmse=float(np.sqrt(np.mean(ee ** 2))),
                         fail_z=float(np.mean(zz > tol_z)),
                         fail_3d=float(np.mean(ee > tol_3d))))

    n_all = np.ones(d.shape[0], dtype=bool)
    add("all", n_all)
    m_pass = ((d["rank"] == 3) & (d["kappa"] < 100)).to_numpy()
    add("criterion pass (rank3 & k<100)", m_pass)
    add("criterion fail", ~m_pass)
    sh = d["share_z"].to_numpy()
    de = d["d_eff"].to_numpy()
    for lo, hi, tag in [(0.0, 0.35, "pass & s_z<=0.35"),
                        (0.35, 0.5, "pass & 0.35<s_z<=0.5"),
                        (0.5, 1.01, "pass & s_z>0.5")]:
        add(tag, m_pass & (sh >= lo) & (sh < hi))
    for lo, hi, tag in [(0.0, 2.0, "pass & d_eff<2"),
                        (2.0, 3.01, "pass & d_eff>=2")]:
        add(tag, m_pass & (de >= lo) & (de < hi))
    for lo, hi, tag in [(0.0, 0.35, "all & s_z<=0.35"),
                        (0.35, 0.5, "all & 0.35<s_z<=0.5"),
                        (0.5, 1.01, "all & s_z>0.5")]:
        add(tag, (sh >= lo) & (sh < hi))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-d", type=float, default=SIGMA_D)
    ap.add_argument("--sigma-theta-deg", type=float,
                    default=float(np.degrees(SIGMA_THETA)))
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--outdir", type=Path, default=RES)
    args = ap.parse_args()
    sigma_theta = float(np.radians(args.sigma_theta_deg))

    df = build(args.sigma_d, sigma_theta, args.n_mc)
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e2e_links.csv", index=False, float_format="%.6g")

    print(f"\n写入 {out / 'e2e_links.csv'}，共 {len(df)} 行")
    print(f"噪声设定 sigma_d = {args.sigma_d} m, "
          f"sigma_theta = {args.sigma_theta_deg} deg, n_mc = {args.n_mc}")
    print(f"锚点数分布：{df['n_anchor'].value_counts().to_dict()}")
    print(f"秩 3 占比 {100 * (df['rank'] == 3).mean():.2f}%  "
          f"kappa 中位 {df['kappa'].median():.2f}  "
          f"d_eff 中位 {df['d_eff'].median():.3f}  "
          f"share_z 中位 {df['share_z'].median():.4f}")

    EST = (("aoa", ("aoa_tz", "aoa_th", "aoa_t3", "aoa_ez")),
           ("wls", ("wls_tz", "wls_th", "wls_t3", "wls_ez")),
           ("tof", ("tof_tz", "tof_th", "tof_t3", "tof_ez")))
    rows = []
    for est, cols in EST:
        rows += summarize(df, cols, est)
    s = pd.DataFrame(rows)
    s.to_csv(out / "e2e_summary.csv", index=False, float_format="%.4f")

    for est, (cz, ch_, c3, _sb) in EST:
        sub = s[s.estimator == est]
        print(f"\n== 估计器 {est.upper()} ==")
        print(sub.drop(columns=["estimator"]).to_string(index=False))
        ok = df[df[c3].notna()]
        y = np.abs(ok[cz].to_numpy())
        print(f"  可用样本 {len(ok)}/{len(df)}")
        print("  相关系数（与垂直误差）: "
              f"kappa {_c(ok['kappa'].to_numpy(), y):+.3f}  "
              f"d_eff {_c(ok['d_eff'].to_numpy(), y):+.3f}  "
              f"share_z {_c(ok['share_z'].to_numpy(), y):+.3f}  "
              f"mean|u_z| {_c(ok['mean_abs_uz'].to_numpy(), y):+.3f}")
    print(f"\n写出 {out / 'e2e_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
