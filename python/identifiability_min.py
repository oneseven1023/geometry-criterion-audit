"""单墙最小可辨识性算例（精确版）

模型（任务书 2.2 节的最小模型）

    二维场景，一面无限长、厚度 d 的墙板占据 x 属于 [0, d]，墙外为自由空间（折射率 1），
    墙内相对介电常数 eps_r 未知（电导率固定，不进入未知量）。
    单站（收发共置）天线位于墙前 x = -h 的一条直线上，横向坐标 u 组成阵列。
    目标为墙后一个点散射体，坐标 p = (x_t, y_t)。

观测为复数基带（频域）信号

    s(f, u) = a * exp(-j 2 pi f tau(u; eta)),    eta = (x_t, y_t, eps_r)^T

tau 为往返时延 = 2 * 单程光学路径 / c，射线按 Snell 定律穿过墙板（无限板，
出射自由空间角等于入射自由空间角）。噪声为复高斯白噪声，方差 sigma^2。

Fisher 信息矩阵

    J = C_J * G,   G = sum_u g_u g_u^T,   g_u = 专 tau(u)/专 eta,
    C_J = 8 pi^2 |a|^2 / sigma^2 * sum_f f^2

因为 C_J 对所有参数是同一正常数，J 的秩、条件数、零空间只由 { g_u } 张成的子空间决定。
也就是说：在该理想模型里，带宽只放大信息量的绝对尺度，不改变可辨识性结构。

位置参数的有效 Fisher 信息（对未知 eps_r 取 Schur 补）

    J_eff = G_pp - G_pth G_thth^{-1} G_thp

关键诊断量是范围信息保留比例

    1 - rho^2,   rho = corr( { 专tau_u/专x_t }, { 专tau_u/专eps_r } )

rho 趋于 1 表示范围方向与介电常数方向在观测上近乎共线，范围信息被未知材料参数吃掉。

输出：results/identifiability_metrics.csv 与 results/figures/ 下的诊断图。
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

C0 = 299_792_458.0  # 光速, m/s


# -----------------------------------------------------------------------------
# 精确的斜入射路径（一维根求解，比 Nelder-Mead 稳定）
# -----------------------------------------------------------------------------
def _angles(th0: float, eps_r: float):
    n = np.sqrt(eps_r)
    s1 = np.clip(np.sin(th0) / n, -1.0, 1.0)
    return th0, float(np.arcsin(s1)), n


def _delta(th0: float, h: float, x_t: float, d: float, eps_r: float) -> float:
    """给定入射自由空间角 th0，射线的横向位移 y_t - y_a。"""
    th0, th1, _ = _angles(th0, eps_r)
    return h * np.tan(th0) + d * np.tan(th1) + (x_t - d) * np.tan(th0)


def _solve_th0(Delta: float, h: float, x_t: float, d: float, eps_r: float) -> float:
    """由横向位移反解入射角，解在 (-pi/2, pi/2) 内唯一。"""
    lim = np.pi / 2 - 1e-12
    f = lambda t: _delta(t, h, x_t, d, eps_r) - Delta
    return float(brentq(f, -lim, lim, xtol=1e-15, rtol=1e-15, maxiter=200))


def oneway_optical_path(y_a: float, x_t: float, y_t: float, d: float, eps_r: float,
                        h: float) -> float:
    """单程光学路径长度（等于 c 乘单程时延）。"""
    th0 = _solve_th0(y_t - y_a, h, x_t, d, eps_r)
    _, th1, n = _angles(th0, eps_r)
    return h / np.cos(th0) + n * d / np.cos(th1) + (x_t - d) / np.cos(th0)


def delay_and_jacobian(y_a: float, eta: np.ndarray, d: float, eps_r_no, h: float):
    """往返时延 tau 与其对 eta = (x_t, y_t, eps_r) 的解析梯度。"""
    x_t, y_t, eps_r = float(eta[0]), float(eta[1]), float(eta[2])
    Delta = y_t - y_a
    th0 = _solve_th0(Delta, h, x_t, d, eps_r)
    _, th1, n = _angles(th0, eps_r)
    s0, s1 = np.sin(th0), np.sin(th1)
    c0, c1 = np.cos(th0), np.cos(th1)
    L0 = x_t - d

    # 总量与其对 th0 的导数
    L = h / c0 + n * d / c1 + L0 / c0
    Lp = (h + L0) * s0 / c0 ** 2 + d * s1 * c0 / c1 ** 3
    Dp = (h + L0) / c0 ** 2 + d * c0 / (n * c1 ** 3)

    # 各参数对 Delta 与 th0 的直接偏导
    dD_dxt = np.tan(th0)
    dL_dxt = 1.0 / c0
    # eps 方向
    den = (eps_r - s0 ** 2)
    dL_deps_th0fix = d * (eps_r - 2 * s0 ** 2) / (2 * den ** 1.5)
    dD_deps = -d * s0 / (2 * den ** 1.5)

    th0_xt = -dD_dxt / Dp
    th0_yt = 1.0 / Dp
    th0_eps = -dD_deps / Dp

    L_xt = dL_dxt + Lp * th0_xt
    L_yt = 0.0 + Lp * th0_yt
    L_eps = dL_deps_th0fix + Lp * th0_eps

    scale = 2.0 / C0  # 往返
    tau = scale * L
    g = scale * np.array([L_xt, L_yt, L_eps])
    return tau, g


def _fd_jacobian(y_a, eta, d, eps_r_no, h, rel=1e-7):
    step = np.array([rel * max(abs(eta[0]), 1.0), rel * max(abs(eta[1]), 1.0),
                     rel * max(abs(eta[2]), 1.0)])
    g = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3); dp[i] = step[i]
        t1 = 2.0 * oneway_optical_path(y_a, *(eta + dp)[:2], d, (eta + dp)[2], h) / C0
        t2 = 2.0 * oneway_optical_path(y_a, *(eta - dp)[:2], d, (eta - dp)[2], h) / C0
        g[i] = (t1 - t2) / (2 * step[i])
    return g


# -----------------------------------------------------------------------------
# 观测与 Fisher 信息
# -----------------------------------------------------------------------------
def fisher_gram(y_list, eta, d, h) -> np.ndarray:
    """G = sum_u g_u g_u^T。"""
    G = np.zeros((3, 3))
    for y_a in y_list:
        _, g = delay_and_jacobian(y_a, eta, d, None, h)
        G += np.outer(g, g)
    return G


def schur_position(G: np.ndarray):
    """对 eps_r 取 Schur 补，返回位置有效信息与诊断量。"""
    G_pp, G_pth, G_thp, G_thth = G[:2, :2], G[:2, 2:3], G[2:3, :2], G[2, 2]
    if G_thth <= 0:
        raise ValueError("eps 方向无信息")
    J_eff = G_pp - G_pth @ G_thp / G_thth
    ev = np.linalg.eigvalsh(J_eff)
    # 范围方向与介电常数方向的相关系数
    gx = np.array([np.nan])  # 由调用方另行计算
    return J_eff, dict(
        eig_min=float(ev[0]), eig_max=float(ev[-1]),
        cond=float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf,
        trace=float(np.trace(J_eff)), trace_Gpp=float(np.trace(G_pp)),
        retention=float(np.trace(J_eff) / np.trace(G_pp)),
    )


def aperture_axis(y_center, length, n_ant):
    if n_ant == 1:
        return np.array([y_center])
    return np.linspace(y_center - length / 2, y_center + length / 2, n_ant)


# -----------------------------------------------------------------------------
# 有限带宽两径可分辨性
# -----------------------------------------------------------------------------
def two_path_fim(bw, dtau, f0, n_f=256):
    f = np.linspace(f0 - bw / 2, f0 + bw / 2, n_f)
    w = f ** 2
    rho = np.sum(w * np.exp(-2j * np.pi * f * dtau)) / np.sum(w)
    F = np.array([[1.0, rho], [np.conj(rho), 1.0]])
    return F, float(abs(rho)), float((1 + abs(rho)) / max(1 - abs(rho), 1e-15)), \
        float(np.min(np.linalg.eigvalsh(F)))


# -----------------------------------------------------------------------------
def run(args):
    d, h = 0.20, 1.00
    eps_true = 6.0
    x_t, y_t = 3.00, 0.50
    f0, bw, n_f = 3.0e9, 2.0e9, 64
    eta = np.array([x_t, y_t, eps_true])

    os.makedirs(args.outdir, exist_ok=True)
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    rows = []

    print("=" * 80)
    print("单墙最小可辨识性算例（精确斜入射模型）")
    print(f"  墙厚 d={d} m, 天线距离 h={h} m, 目标=({x_t},{y_t}) m, eps_r={eps_true}")
    print(f"  频段 f0={f0/1e9} GHz, B={bw/1e9} GHz, 频点数 {n_f}")
    print("=" * 80)

    # 解析 vs 有限差分一致性自检
    _, gA = delay_and_jacobian(y_t, eta, d, None, h)
    gF = _fd_jacobian(y_t, eta, d, None, h)
    print("\n[0] 解析雅可比与有限差分一致性（单站，天线与目标对齐）")
    print(f"    解析g = {np.array2string(gA, precision=4, floatmode='maxprec')}")
    print(f"    差分g = {np.array2string(gF, precision=4, floatmode='maxprec')}")
    relerr = np.max(np.abs(gA - gF) / np.maximum(np.abs(gF), 1e-30))
    print(f"    最大相对偏差 = {relerr:.2e}    (确认导数实现正确)")

    # 用非对齐天线再做一次校验（避免对称点退化）
    _, gA2 = delay_and_jacobian(y_t + 0.7, eta, d, None, h)
    gF2 = _fd_jacobian(y_t + 0.7, eta, d, None, h)
    print(f"    非对齐天线: 解析 {np.array2string(gA2, precision=4, floatmode='maxprec')}")
    print(f"                差分 {np.array2string(gF2, precision=4, floatmode='maxprec')}")
    print(f"                最大相对偏差 = {np.max(np.abs(gA2-gF2)/np.maximum(np.abs(gF2),1e-30)):.2e}")

    # ---------- [1] 角孔径扫描 ----------
    print("\n[1] 秩、条件数与范围信息保留比例随阵列角孔径变化（9 天线，居中对称）")
    print(f"{'阵列长(m)':>9s} {'角孔径(°)':>9s} {'rank(G)':>7s} {'sig_min(G)':>11s} "
          f"{'rank(Je)':>8s} {'eigmin(Je)':>11s} {'cond(Je)':>10s} {'1-rho^2':>9s} {'保留比':>8s}")
    for L in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]:
        y_list = aperture_axis(y_t, L, 1 if L == 0 else 9)
        G = fisher_gram(y_list, eta, d, h)
        sv = np.linalg.svd(G, compute_uv=False)
        sv0 = max(sv[0], 1e-300)
        rankG = int(np.sum(sv > sv0 * 1e-9))
        J_eff, dg = schur_position(G)
        ev = np.linalg.eigvalsh(J_eff)
        rankE = int(np.sum(ev > max(ev[-1], 1e-300) * 1e-9))
        # 相关系数 rho 于 x_t 与 eps 方向
        gxs, gps = [], []
        for y_a in y_list:
            _, g = delay_and_jacobian(y_a, eta, d, None, h)
            gxs.append(g[0]); gps.append(g[2])
        gxs, gps = np.array(gxs), np.array(gps)
        rho = abs(np.sum(gxs * gps) / np.sqrt(np.sum(gxs ** 2) * np.sum(gps ** 2)))
        ang = 2 * np.degrees(np.arctan((L / 2) / (h + x_t)))
        print(f"{L:9.3f} {ang:9.3f} {rankG:7d} {sv[-1]:11.3e} {rankE:8d} "
              f"{ev[0]:11.3e} {dg['cond']:10.3e} {1-rho**2:9.3e} {dg['retention']:8.3f}")
        rows.append(dict(kind="aperture_centered", param=L, n_ant=len(y_list),
                         angular_extent_deg=float(ang), rank_G=rankG, sigma_min_G=float(sv[-1]),
                         cond_G=float(sv0 / max(sv[-1], 1e-300)), rank_Geff=rankE, eig_min_Geff=float(ev[0]),
                         cond_Geff=dg["cond"], one_minus_rho2=float(1 - rho ** 2),
                         retention=dg["retention"]))

    # ---------- [2] 目标偏离阵列中心（打破 y 对称） ----------
    print("\n[2] 目标相对阵列中心横向偏置 0.5 m（打破 y 对称），9 天线")
    print(f"{'阵列长(m)':>9s} {'rank(G)':>7s} {'rank(Je)':>8s} {'eigmin(Je)':>11s} "
          f"{'cond(Je)':>10s} {'1-rho^2':>9s}")
    for L in [0.0, 0.2, 0.4, 0.8, 1.6, 3.2]:
        y_list = aperture_axis(y_t - 0.5, L, 1 if L == 0 else 9)
        G = fisher_gram(y_list, eta, d, h)
        sv = np.linalg.svd(G, compute_uv=False)
        sv0 = max(sv[0], 1e-300)
        rankG = int(np.sum(sv > sv0 * 1e-9))
        J_eff, dg = schur_position(G)
        ev = np.linalg.eigvalsh(J_eff)
        rankE = int(np.sum(ev > max(ev[-1], 1e-300) * 1e-9))
        gxs, gps = [], []
        for y_a in y_list:
            _, g = delay_and_jacobian(y_a, eta, d, None, h)
            gxs.append(g[0]); gps.append(g[2])
        gxs, gps = np.array(gxs), np.array(gps)
        rho = abs(np.sum(gxs * gps) / np.sqrt(np.sum(gxs ** 2) * np.sum(gps ** 2)))
        print(f"{L:9.3f} {rankG:7d} {rankE:8d} {ev[0]:11.3e} {dg['cond']:10.3e} {1-rho**2:9.3e}")
        rows.append(dict(kind="aperture_offset", param=L, n_ant=len(y_list),
                         angular_extent_deg=np.nan, rank_G=rankG, sigma_min_G=float(sv[-1]),
                         cond_G=float(sv0 / max(sv[-1], 1e-300)), rank_Geff=rankE, eig_min_Geff=float(ev[0]),
                         cond_Geff=dg["cond"], one_minus_rho2=float(1 - rho ** 2),
                         retention=dg["retention"]))

    # ---------- [3] 天线数阈值 ----------
    print("\n[3] 固定孔径 0.8 m，天线数对可辨识性的影响")
    print(f"{'天线数':>6s} {'rank(G)':>7s} {'sig_min(G)':>11s} {'rank(Je)':>8s} {'eigmin(Je)':>11s}")
    for n in [1, 2, 3, 4, 6, 9, 17]:
        y_list = aperture_axis(y_t, 0.8, n)
        G = fisher_gram(y_list, eta, d, h)
        sv = np.linalg.svd(G, compute_uv=False)
        sv0 = max(sv[0], 1e-300)
        rankG = int(np.sum(sv > sv0 * 1e-9))
        J_eff, dg = schur_position(G)
        ev = np.linalg.eigvalsh(J_eff)
        rankE = int(np.sum(ev > max(abs(ev[-1]), 1e-300) * 1e-9))
        print(f"{n:6d} {rankG:7d} {sv[-1]:11.3e} {rankE:8d} {ev[0]:11.3e}")
        rows.append(dict(kind="n_ant", param=n, n_ant=n, angular_extent_deg=np.nan,
                         rank_G=rankG, sigma_min_G=float(sv[-1]), cond_G=float(sv0/max(sv[-1],1e-300)),
                         rank_Geff=rankE, eig_min_Geff=float(ev[0]), cond_Geff=dg["cond"],
                         one_minus_rho2=np.nan, retention=dg["retention"]))

    # ---------- [4] 单站零空间 ----------
    print("\n[4] 单站零空间：位置与介电常数的不可区分方向")
    _, g1 = delay_and_jacobian(y_t, eta, d, None, h)
    print(f"    g = (专tau/专x_t, 专tau/专y_t, 专tau/专eps_r) = "
          f"({g1[0]:+.4e}, {g1[1]:+.4e}, {g1[2]:+.4e})")
    _, _, vt = np.linalg.svd(g1.reshape(1, 3))
    for i in range(1, 3):
        v = vt[i] / np.max(np.abs(vt[i]))
        print(f"    零空间方向{i}: (dx_t, dy_t, d_eps_r) = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})")
    print("    含义：沿零空间方向同时改变目标坐标与介电常数，往返时延一阶不变，")
    print("          单站观测无法把两者分开。")

    # ---------- [5] 带宽两径可分辨性 ----------
    print("\n[5] 有限带宽下两条接近路径的可分辨阈值（等幅两径）")
    print(f"{'dtau(ns)':>9s} {'B*dtau':>8s} {'|rho|':>8s} {'cond(FIM_tau)':>14s} {'eig_min':>10s}")
    for dtau_ns in [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        dtau = dtau_ns * 1e-9
        _, rho_abs, cond, emin = two_path_fim(bw, dtau, f0, n_f)
        print(f"{dtau_ns:9.3f} {bw*dtau:8.3f} {rho_abs:8.4f} {cond:14.3e} {emin:10.3e}")
        rows.append(dict(kind="two_path", param=dtau_ns, n_ant=np.nan,
                         angular_extent_deg=np.nan, rank_G=np.nan, sigma_min_G=np.nan,
                         cond_G=np.nan, rank_Geff=np.nan, eig_min_Geff=np.nan,
                         cond_Geff=cond, one_minus_rho2=float(1 - rho_abs ** 2),
                         retention=emin))

    # ---------- [6] 加入墙前表面回波后的范围信息保留 ----------
    print("\n[6] 加入墙前表面回波（直接观测 eps_r）后，范围信息保留比例随辅助信息强度变化")
    print("    模型：G_ee = G_ee^target * (1 + beta)，beta 为墙回波对 eps_r 的独立信息占比")
    print("    范围方向保留 = 1 - rho_t^2 / (1 + beta)")
    L6 = 1.6
    y_list6 = aperture_axis(y_t, L6, 9)
    G6 = fisher_gram(y_list6, eta, d, h)
    Gxx, Gxe, Gee = G6[0, 0], G6[0, 2], G6[2, 2]
    rho_t2 = Gxe ** 2 / (Gxx * Gee)
    print(f"    仅目标回波时 rho_t^2 = {rho_t2:.6f}  (即范围信息保留 {1-rho_t2:.3e})")
    print(f"{'beta':>9s} {'范围信息保留':>14s} {'说明':>28s}")
    for beta in [0.0, 0.1, 1.0, 10.0, 100.0, 1e6]:
        ret = 1 - rho_t2 / (1 + beta)
        note = "只有目标回波" if beta == 0 else ("墙回波占优" if beta >= 10 else "")
        print(f"{beta:9.3g} {ret:14.6f} {note:>28s}")
        rows.append(dict(kind="wall_echo_retention", param=beta, n_ant=9,
                         angular_extent_deg=np.nan, rank_G=np.nan, sigma_min_G=np.nan,
                         cond_G=np.nan, rank_Geff=np.nan, eig_min_Geff=np.nan,
                         cond_Geff=np.nan, one_minus_rho2=rho_t2, retention=float(ret)))

    keys = ["kind", "param", "n_ant", "angular_extent_deg", "rank_G", "sigma_min_G",
            "cond_G", "rank_Geff", "eig_min_Geff", "cond_Geff", "one_minus_rho2", "retention"]
    csv_path = os.path.join(args.outdir, "identifiability_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n已写出 {csv_path}")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 尝试启用中文字体，失败则退回英文标签
        cand = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
        have_cjk = False
        for f in cand:
            try:
                font_manager.findfont(f, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [f]
                plt.rcParams["axes.unicode_minus"] = False
                have_cjk = True
                break
            except Exception:
                continue
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
                             "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
        L_ = "阵列角孔径 (deg)" if have_cjk else "Aperture angular extent (deg)"
        T1 = "位置有效信息条件数" if have_cjk else "Condition number of J_eff"
        T2 = "范围信息保留比例 1-rho^2" if have_cjk else "Retained range info 1-rho^2"
        T3 = "有限带宽两径可分辨性" if have_cjk else "Two-path resolvability"

        ce = [r for r in rows if r["kind"] == "aperture_centered"]
        angs = [r["angular_extent_deg"] for r in ce]
        conds = [r["cond_Geff"] for r in ce]
        ret = [r["one_minus_rho2"] for r in ce]
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
        ax[0].semilogy(angs, conds, "o-", lw=2, ms=7, color="#c0392b")
        ax[0].set_xlabel(L_); ax[0].set_ylabel(T1); ax[0].set_title(T1); ax[0].grid(alpha=0.35)
        ax[1].semilogy(angs, ret, "s-", lw=2, ms=7, color="#2c6fbb")
        ax[1].set_xlabel(L_); ax[1].set_ylabel(T2); ax[1].set_title(T2); ax[1].grid(alpha=0.35)
        fig.tight_layout()
        p1 = os.path.join(figdir, "fig_identifiability_aperture.png")
        fig.savefig(p1, dpi=150); plt.close(fig)

        tp = [r for r in rows if r["kind"] == "two_path"]
        xb = [bw * r["param"] * 1e-9 for r in tp]
        yc = [r["cond_Geff"] for r in tp]
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.loglog(xb, yc, "^-", lw=2, ms=8, color="#8e44ad")
        ax.axvline(1.0, color="gray", ls="--", lw=1.5)
        ax.text(1.06, min(yc) * 2, "B*dtau = 1", fontsize=11, color="gray")
        ax.set_xlabel("B * dtau"); ax.set_ylabel("cond(FIM_tau)"); ax.set_title(T3)
        ax.grid(alpha=0.35, which="both")
        fig.tight_layout()
        p2 = os.path.join(figdir, "fig_two_path_resolvability.png")
        fig.savefig(p2, dpi=150); plt.close(fig)
        print(f"已写出图: {p1}\n已写出图: {p2}   (中文字体={'可用' if have_cjk else '不可用，已用英文标签'})")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")

    print("\n完成。")


def main():
    ap = argparse.ArgumentParser(description="单墙最小可辨识性算例")
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
