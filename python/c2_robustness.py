#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_robustness.py -- 判据侧诊断量的三项稳定性复核

要回答的三个问题
----------------
1. 数据集给出的是射线追踪的到达方向真值，而真实接收机只能得到带误差的角度。
   若把到达方向按角度误差 sigma_theta 扰动，诊断量（s_z、kappa、d_eff）与
   「方向性误导通过」这一判定还站得住吗？
2. 论文的 FIM 用全部 MPC，端到端估计器只用每信道首径。只保留首径时诊断量
   是否等价？若等价，则审计与估计器依赖的是同一份信息。
3. 式 (3) 的小特征值下限 1e-12 换成 1e-8 至 1e-16，诊断量与判定是否稳定？

为什么只扰动方向
----------------
位置 FIM 只通过单位方向 u_i 与权重 w_i 依赖几何，时延本身不进入 J。因此阵列
角度误差是唯一能改变审计结论的测量误差；时延误差只影响估计器，不影响判据侧。

口径
----
与 c2_fullscale_vec 逐位一致：动态范围 30 dB、按接收机真实坐标分组、
特征值相对下限 1e-12、秩门限 1e-9、权重 10**(g/5)。

输出
----
results/robust_dirnoise.csv     逐 split、逐 sigma 的汇总
results/robust_firstpath.csv    首径 FIM 与全 MPC FIM 的对照
results/robust_epsfloor.csv     特征值下限扫描
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

EPS0 = 1e-12
RANK_REL = 1e-9
DR_DB = 30.0
SIGMAS_DEG = (2.0, 5.0, 10.0)
EPS_LIST = (1e-8, 1e-10, 1e-12, 1e-14, 1e-16)


# --------------------------------------------------------------------------- #
# 由 (lam, Q) 直接给出指标，便于扫描特征值下限而不重做特征分解
# --------------------------------------------------------------------------- #
def metrics_from_eig(lam: np.ndarray, Q: np.ndarray, eps: float) -> dict:
    lam_max = lam[:, -1]
    ok = lam_max > 0
    lam_c = np.where(ok[:, None],
                     np.clip(lam, lam_max[:, None] * eps, None), 1.0)
    inv = 1.0 / lam_c
    var_z = (Q[:, 2, :] ** 2 * inv).sum(axis=1)
    trace_inv = inv.sum(axis=1)
    safe = np.where(trace_inv > 0, trace_inv, 1.0)
    share_z = np.where(trace_inv > 0, var_z / safe, 1.0)
    kappa = np.where(ok, lam_c[:, -1] / lam_c[:, 0], np.inf)
    d_eff = (lam.sum(axis=1) ** 2) / np.maximum((lam ** 2).sum(axis=1), 1e-300)
    rank = (lam > (lam_max[:, None] * RANK_REL)).sum(axis=1)
    return dict(share_z=share_z, kappa=kappa, d_eff=d_eff, rank=rank)


def eig_batch(J: np.ndarray):
    J = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    lam, Q = np.linalg.eigh(J)
    lam = np.clip(lam, 0.0, None)
    return lam, Q


def fim_from(dirs: np.ndarray, w: np.ndarray, grp: np.ndarray, ngrp: int) -> np.ndarray:
    """按组号累加 w * u u^T，返回 (ngrp,3,3)。要求 dirs 已按 grp 排序。"""
    outer = (dirs[:, :, None] * dirs[:, None, :]) * w[:, None, None]
    st = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    J = np.add.reduceat(outer.reshape(dirs.shape[0], 9), st, axis=0).reshape(-1, 3, 3)
    return J, st


def process_floor(f: h5py.File, floor_id: int, sigmas: np.ndarray):
    bounds = wio.floor_channel_bounds(f)
    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    if c1 <= c0:
        return None
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    tx_idx, rx_idx, pstart = ch[:, 0], ch[:, 1], ch[:, 2]
    M = ch.shape[0]

    utx, urx = np.unique(tx_idx), np.unique(rx_idx)
    tx_xyz = np.asarray(f["tx"][utx], dtype=np.float64)[np.searchsorted(utx, tx_idx)]
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
    rx_uid = np.asarray(rx_uid).ravel()
    G = int(rx_uid.max()) + 1

    grp_p = rx_uid[ch_of_path]
    w = 10.0 ** (paths[:, 0] / wio.GAIN_DB_DIVISOR)
    aoa_az, zoa = paths[:, 5], paths[:, 6]

    order = np.argsort(grp_p, kind="stable")
    grp_s = grp_p[order]
    starts = np.flatnonzero(np.r_[True, grp_s[1:] != grp_s[:-1]])
    cnt_all = np.diff(np.append(starts, K))
    w_s = w[order]
    g_max = np.maximum.reduceat(w_s, starts)
    keep = w_s >= (np.repeat(g_max, cnt_all) * 10.0 ** (-DR_DB / 10.0))

    az_o = aoa_az[order][keep]
    z_o = zoa[order][keep]
    w_o = w_s[keep] / np.repeat(g_max, cnt_all)[keep]
    grp_o = grp_s[keep]

    is_first = np.zeros(K, dtype=bool)
    is_first[rel] = True
    first_o = is_first[order][keep]

    rng = np.random.default_rng(74112)

    def run(az, z, ww, gg):
        d = np.stack([np.sin(z) * np.cos(az),
                      np.sin(z) * np.sin(az),
                      np.cos(z)], axis=1)
        J, st = fim_from(d, ww, gg, G)
        lam, Q = eig_batch(J)
        return lam, Q, st

    # ---- 基线 ----
    lam0, Q0, st0 = run(az_o, z_o, w_o, grp_o)
    base = metrics_from_eig(lam0, Q0, EPS0)
    gid = grp_o[st0]

    # ---- 仅首径 ----
    lam1, Q1, st1 = run(az_o[first_o], z_o[first_o], w_o[first_o], grp_o[first_o])
    fp = metrics_from_eig(lam1, Q1, EPS0)
    gid1 = grp_o[first_o][st1]
    fp_sel = np.isin(gid, gid1)
    fp_z = np.full(G, np.nan)
    fp_k = np.full(G, np.nan)
    fp_d = np.full(G, np.nan)
    fp_verdict = np.zeros(G, dtype=bool)
    fp_z[gid1] = fp["share_z"]
    fp_k[gid1] = fp["kappa"]
    fp_d[gid1] = fp["d_eff"]
    fp_verdict[gid1] = (fp["rank"] == 3) & (fp["kappa"] < 100) & (fp["share_z"] > 0.5)

    # ---- 方向误差 ----
    noise = []
    for sg in sigmas:
        az_n = az_o + rng.normal(0.0, sg, az_o.size)
        z_n = np.clip(z_o + rng.normal(0.0, sg, z_o.size), 1e-6, np.pi - 1e-6)
        lam_n, Q_n, _ = run(az_n, z_n, w_o, grp_o)
        noise.append(metrics_from_eig(lam_n, Q_n, EPS0))

    # ---- 特征值下限 ----
    epsm = []
    for e in EPS_LIST:
        epsm.append(metrics_from_eig(lam0, Q0, e))

    return dict(gid=gid, lam=lam0, Q=Q0,
                base=base, noise=noise, eps=epsm,
                fp_sel=fp_sel, fp_z=fp_z[gid], fp_k=fp_k[gid],
                fp_d=fp_d[gid], fp_verdict=fp_verdict[gid])


def verdict(m: dict) -> np.ndarray:
    return (m["rank"] == 3) & (m["kappa"] < 100) & (m["share_z"] > 0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--floor-limit", type=int, default=0)
    args = ap.parse_args()

    sig_rad = np.radians(np.array(SIGMAS_DEG))
    splits = wio.discover_splits(ROOT)
    if args.split:
        splits = [s for s in splits if s.tag == args.split]
    RES.mkdir(parents=True, exist_ok=True)

    rows_dn, rows_fp, rows_ep = [], [], []
    pool = {"sz": [], "k": [], "v": [], "z0": [],
            "fp_sz": [], "fp_k": [], "fp_v": [],
            "dn_sz": [[] for _ in SIGMAS_DEG], "dn_fz": [[] for _ in SIGMAS_DEG],
            "dn_fa": [[] for _ in SIGMAS_DEG], "dn_dz": [[] for _ in SIGMAS_DEG],
            "ep_fl": [[] for _ in EPS_LIST]}
    t_all = time.time()
    for info in splits:
        t0 = time.time()
        acc = {"base_sz": [], "base_k": [], "base_v": [], "base_z0": [],
               "fp_sz": [], "fp_k": [], "fp_v": [],
               "eps_sz": [[] for _ in EPS_LIST],
               "eps_k": [[] for _ in EPS_LIST], "eps_v": [[] for _ in EPS_LIST],
               "dn_sz": [[] for _ in SIGMAS_DEG], "dn_k": [[] for _ in SIGMAS_DEG],
               "dn_v": [[] for _ in SIGMAS_DEG], "dn_ez": [[] for _ in SIGMAS_DEG],
               "dn_fz": [[] for _ in SIGMAS_DEG], "dn_fa": [[] for _ in SIGMAS_DEG]}
        nlink = 0
        with h5py.File(info.path, "r") as f:
            floors = wio.list_floors(f)
            if args.floor_limit:
                floors = floors[: args.floor_limit]
            for fid, fidx in floors:
                r = process_floor(f, fid, sig_rad)
                if r is None:
                    continue
                b = r["base"]
                v0 = verdict(b)
                z0 = b["share_z"] > 0.5
                nlink += b["share_z"].size
                acc["base_sz"].append(b["share_z"])
                acc["base_k"].append(b["kappa"])
                acc["base_v"].append(v0)
                acc["base_z0"].append(z0)
                acc["fp_sz"].append(r["fp_z"])
                acc["fp_k"].append(r["fp_k"])
                acc["fp_v"].append(r["fp_verdict"])
                for j, m in enumerate(r["eps"]):
                    acc["eps_sz"][j].append(m["share_z"])
                    acc["eps_k"][j].append(m["kappa"])
                    acc["eps_v"][j].append(verdict(m) != v0)
                for j, m in enumerate(r["noise"]):
                    acc["dn_sz"][j].append(m["share_z"])
                    acc["dn_k"][j].append(m["kappa"])
                    acc["dn_v"][j].append(verdict(m) != v0)
                    acc["dn_fz"][j].append((m["share_z"] > 0.5) != z0)
                    acc["dn_fa"][j].append(verdict(m) != v0)
                    acc["dn_ez"][j].append(np.abs(m["share_z"] - b["share_z"]))
        if nlink == 0:
            continue

        def cat(key, j=None):
            v = acc[key] if j is None else acc[key][j]
            return np.concatenate(v)

        sz, kp, v0, z0 = cat("base_sz"), cat("base_k"), cat("base_v"), cat("base_z0")
        tag = info.tag
        pool["sz"].append(sz.astype(np.float32))
        pool["k"].append(kp.astype(np.float32))
        pool["v"].append(v0)
        pool["z0"].append(z0)
        rows_dn.append(dict(split=tag, n=nlink,
                            sz_med=float(np.median(sz)), k_med=float(np.median(kp)),
                            sz_gt=float(np.mean(z0)), mislead=float(np.mean(v0))))

        fp_sz, fp_kp, fp_v = cat("fp_sz"), cat("fp_k"), cat("fp_v")
        ok = np.isfinite(fp_sz)
        pool["fp_sz"].append(fp_sz[ok].astype(np.float32))
        pool["fp_k"].append(fp_kp[ok].astype(np.float32))
        pool["fp_v"].append(fp_v[ok])
        rows_fp.append(dict(split=tag, n=int(ok.sum()),
                            sz_all_med=float(np.median(sz[ok])),
                            sz_fp_med=float(np.median(fp_sz[ok])),
                            k_all_med=float(np.median(kp[ok])),
                            k_fp_med=float(np.median(fp_kp[ok])),
                            rho=float(np.corrcoef(sz[ok], fp_sz[ok])[0, 1]),
                            rho_spear=float(np.corrcoef(
                                np.argsort(np.argsort(sz[ok])),
                                np.argsort(np.argsort(fp_sz[ok])))[0, 1]),
                            mislead_all=float(np.mean(v0[ok])),
                            mislead_fp=float(np.mean(fp_v[ok])),
                            agree=float(np.mean(v0[ok] == fp_v[ok]))))

        for j, e in enumerate(EPS_LIST):
            fl = cat("eps_v", j)
            pool["ep_fl"][j].append(fl)
            rows_ep.append(dict(split=tag, eps=e,
                                sz_med=float(np.median(cat("eps_sz", j))),
                                k_med=float(np.median(cat("eps_k", j))),
                                flip=float(np.mean(fl))))

        for j, sg in enumerate(SIGMAS_DEG):
            dn_sz, fz, fa, dz = (cat("dn_sz", j), cat("dn_fz", j),
                                 cat("dn_fa", j), cat("dn_ez", j))
            pool["dn_sz"][j].append(dn_sz.astype(np.float32))
            pool["dn_fz"][j].append(fz)
            pool["dn_fa"][j].append(fa)
            pool["dn_dz"][j].append(dz.astype(np.float32))
            rows_dn.append(dict(split=tag, n=nlink, sigma_deg=sg,
                                sz_med=float(np.median(dn_sz)),
                                k_med=float(np.median(cat("dn_k", j))),
                                sz_gt=float(np.mean(dn_sz > 0.5)),
                                flip_share=float(np.mean(fz)),
                                flip_any=float(np.mean(fa)),
                                dz_med=float(np.median(dz)),
                                baseline_sz_med=float(np.median(sz)),
                                baseline_k_med=float(np.median(kp)),
                                baseline_sz_gt=float(np.mean(z0)),
                                baseline_mislead=float(np.mean(v0))))
        print(f"[{tag}] {nlink} links, {time.time()-t0:.1f}s", flush=True)

    # ---- 汇总全量 ----
    if pool["sz"]:
        S = np.concatenate(pool["sz"]); K = np.concatenate(pool["k"])
        V = np.concatenate(pool["v"]); Z = np.concatenate(pool["z0"])
        rows_dn.append(dict(split="ALL", n=int(S.size),
                            sz_med=float(np.median(S)), k_med=float(np.median(K)),
                            sz_gt=float(np.mean(Z)), mislead=float(np.mean(V))))
        for j, sg in enumerate(SIGMAS_DEG):
            dn = np.concatenate(pool["dn_sz"][j])
            rows_dn.append(dict(split="ALL", n=int(S.size), sigma_deg=sg,
                                sz_med=float(np.median(dn)),
                                sz_gt=float(np.mean(dn > 0.5)),
                                flip_share=float(np.mean(np.concatenate(pool["dn_fz"][j]))),
                                flip_any=float(np.mean(np.concatenate(pool["dn_fa"][j]))),
                                dz_med=float(np.median(np.concatenate(pool["dn_dz"][j]))),
                                baseline_sz_med=float(np.median(S)),
                                baseline_sz_gt=float(np.mean(Z)),
                                baseline_mislead=float(np.mean(V))))
        for j, e in enumerate(EPS_LIST):
            rows_ep.append(dict(split="ALL", eps=e,
                                flip=float(np.mean(np.concatenate(pool["ep_fl"][j])))))
        fs = np.concatenate(pool["fp_sz"]); fk = np.concatenate(pool["fp_k"])
        fv = np.concatenate(pool["fp_v"])
        ok = np.isfinite(fs)
        rows_fp.append(dict(split="ALL", n=int(ok.sum()),
                            sz_all_med=float(np.median(S[ok])),
                            sz_fp_med=float(np.median(fs[ok])),
                            k_all_med=float(np.median(K[ok])),
                            k_fp_med=float(np.median(fk[ok])),
                            rho=float(np.corrcoef(S[ok], fs[ok])[0, 1]),
                            mislead_all=float(np.mean(V[ok])),
                            mislead_fp=float(np.mean(fv[ok])),
                            agree=float(np.mean(V[ok] == fv[ok]))))

    def dump(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)

    dump(RES / "robust_dirnoise.csv", rows_dn)
    dump(RES / "robust_firstpath.csv", rows_fp)
    dump(RES / "robust_epsfloor.csv", rows_ep)
    print(f"total {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
