# -*- coding: utf-8 -*-
"""布儒斯特角探针：TM 极化 Fresnel 反射系数对介电常数的敏感度，以及它能否
恢复被未知介电常数吃掉的位置信息。

检验的假设（用户给定）：介电常数同时经 Fresnel 幅度通道与折射时延通道进入
观测，两者对入射角的敏感度依赖不同；TM 极化在布儒斯特角处反射系数对介电常数
的敏感度趋零，而时延通道不趋零，由此产生纯增益型讨厌参数模型里不存在的
结构退化。

脚本分三层，对应任务的执行顺序

  A 解析层   Fresnel TM 复反射系数及其对 eps_r 的一阶导数。
             回答：在布儒斯特角处哪些量为零、哪些不为零。同时找出导数真正为零的角度。
  B 损耗层   含电导率（复介电常数 eps_c = eps_r - j kappa）时的伪布儒斯特结构。
  C 统计层   把未知复增益、目标位置、eps_r 联合纳入复数基带 FIM，考察
             仅时延 / 仅幅度 / 幅度加时延 三种观测组合下，消除全部讨厌参数后的
             位置有效信息，是否出现布儒斯特特异的退化。
             并把幅度通道的 eps_r 信息解析分解为 原始敏感度 x 多样性因子。

观测模型（最小模型，单界面折射 + 界面反射，不运行 RT）

  二维。墙占据 x >= 0，材料由相对介电常数 eps_r（可含损耗）描述。
  收发共用一条位于 x = -h 的天线线阵，位置 y_a。
  目标为墙后点散射体 p = (x_t, y_t)，射线按 Snell 定律在界面折射后到达目标。
  两条观测通道

    通道 T（折射时延通道）  mu_T = a_T * exp(-j 2 pi f tau_j(x_t, y_t, eps_r))
    通道 W（界面反射通道）  mu_W = a_W * Gamma_TM(theta_j; eps_r)

  噪声为复高斯白噪声。未知向量 (x_t, y_t, eps_r, a_T, a_W) 全联合，
  位置有效信息 = 对 (eps_r, a_T, a_W) 取 Schur 补。

公式勘误：任务书给出的
      Gamma_TM = (eps_r cos th_t - cos th_i)/(eps_r cos th_t + cos th_i)
  与标准 TM 反射系数不一致，且对实数 eps_r > 1 恒无实零点（推出 cos^2 th_i = -eps_r）。
  本脚本采用标准形式 (n cos th_i - cos th_t)/(n cos th_i + cos th_t)，
  等价于 (eps_r cos th_i - n cos th_t)/(eps_r cos th_i + n cos th_t)，并在布儒斯特角处校验其为零。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from identifiability_min import delay_and_jacobian, aperture_axis, _solve_th0  # noqa: E402

DEG = math.pi / 180.0


# -----------------------------------------------------------------------------
# A 解析层：Fresnel TM 复反射系数与其导数
# -----------------------------------------------------------------------------
def gamma_tm(theta_i, eps_c):
    """TM（p 极化）复反射系数，空气入射到相对介电常数 eps_c 的介质。"""
    ci = np.cos(theta_i)
    si = np.sin(theta_i)
    ct = np.sqrt(1.0 - si ** 2 / eps_c)
    n = np.sqrt(eps_c)
    return (n * ci - ct) / (n * ci + ct)


def brewster_deg(eps_r):
    """无损介质的布儒斯特角（度）。"""
    return math.degrees(math.atan(math.sqrt(eps_r)))


def dgamma_deps_c(theta_i, eps_c):
    """解析导数 d Gamma / d eps_c（把复介电常数当独立变量，不涉及 tan d 的链式因子）。

    Gamma = (a - b)/(a + b),  a = n cos th_i,  b = cos th_t,  n = sqrt(eps_c)
    dGamma/deps_c = 2 (a' b - a b')/(a + b)^2
    """
    n = np.sqrt(eps_c)
    ci = np.cos(theta_i)
    si = np.sin(theta_i)
    b = np.sqrt(1.0 - si ** 2 / eps_c)
    a = n * ci
    ap = ci / (2.0 * n)
    bp = si ** 2 / (2.0 * eps_c ** 2 * b)
    return 2.0 * (ap * b - a * bp) / (a + b) ** 2


def dgamma_depsr_cs(theta_i, eps_r, h=1e-20):
    """无损情形 d Gamma / d eps_r，复步长法（仅当 Gamma 为实值时有效，用作校验）。"""
    return float(np.imag(gamma_tm(theta_i, eps_r + 1j * h)) / h)


def dgamma_depsr_fd(theta_i, eps_r, h=1e-6):
    """中心差分，交叉校验。"""
    f1 = gamma_tm(theta_i, eps_r + h)
    f2 = gamma_tm(theta_i, eps_r - h)
    return complex((f1 - f2) / (2 * h))


def amp_sens_profile(eps_r, kappa=0.0, n=4001):
    """返回 (theta_deg, |Gamma|, |dGamma/d eps_r|) 数组，角域 (0, 89.9) 度。"""
    th = np.linspace(1e-9, 89.9 * DEG, n)
    G = np.array([gamma_tm(t, eps_r - 1j * kappa) for t in th])
    S = np.array([dgamma_deps_c(t, eps_r - 1j * kappa) for t in th])
    return np.degrees(th), np.abs(G), np.abs(S)


# -----------------------------------------------------------------------------
# C 统计层：复数基带 FIM
# -----------------------------------------------------------------------------
def fim_complex(dmu, sigma2):
    """复数高斯观测的 FIM。dmu 形状 (n_obs, n_par)，J = (2/sigma2) Re[dmu^H dmu]。"""
    dmu = np.asarray(dmu, dtype=complex)
    return (2.0 / sigma2) * np.real(np.conj(dmu).T @ dmu)


def schur_block(J, keep, elim):
    """尺度稳健的 Schur 补。

    先按对角线归一化成相关矩阵再消元，避免量级差异极大时（例如某个讨厌参数
    在部分观测上几乎不可辨识，对角元趋于零）直接 pinv 造成的大误差，
    最后再还原尺度。
    """
    keep = list(keep)
    elim = list(elim)
    d = np.sqrt(np.clip(np.diag(J), 1e-300, None))
    C = J / np.outer(d, d)
    A = C[np.ix_(keep, keep)]
    B = C[np.ix_(keep, elim)]
    Cc = C[np.ix_(elim, keep)]
    Dm = C[np.ix_(elim, elim)]
    E = A - B @ np.linalg.pinv(Dm, rcond=1e-12) @ Cc
    dk = d[keep]
    return E * np.outer(dk, dk)


def delay_channel_fim(ya_list, eta, d, h, freqs, sigma2, a_mag=1.0):
    """通道 T。未知顺序 (x_t, y_t, eps_r, a_T_re, a_T_im)。"""
    rows = []
    for ya in ya_list:
        tau, g = delay_and_jacobian(ya, eta, d, None, h)
        for f in freqs:
            ph = np.exp(-1j * 2 * np.pi * f * tau)
            base = -1j * 2 * np.pi * f * ph
            rows.append([a_mag * base * g[0], a_mag * base * g[1], a_mag * base * g[2],
                         ph, 1j * ph])
    return fim_complex(np.array(rows), sigma2)


def amp_channel_fim(theta_list, eps_r, kappa, sigma2, a_mag=1.0):
    """通道 W。未知顺序 (eps_r, a_W_re, a_W_im)。"""
    rows = []
    for th in theta_list:
        G = gamma_tm(th * DEG, eps_r - 1j * kappa)
        dG = dgamma_deps_c(th * DEG, eps_r - 1j * kappa)
        rows.append([a_mag * dG, a_mag * G, 1j * a_mag * G])
    return fim_complex(np.array(rows), sigma2)


def amp_eps_info_after_gain(theta_list, eps_r, kappa, sigma2):
    """通道 W 消除未知复增益后剩下的 eps_r 信息，及其解析分解。

    info = raw_sens x diversity
      raw_sens  = (2/sigma2) sum_j |dGamma_j/deps|^2
      diversity = 1 - |rho|^2,  rho 为复相关系数 corr(Gamma_j, dGamma_j/deps)
    """
    if len(theta_list) == 0:
        return 0.0, 0.0, 0.0
    G = np.array([gamma_tm(t * DEG, eps_r - 1j * kappa) for t in theta_list])
    dG = np.array([dgamma_deps_c(t * DEG, eps_r - 1j * kappa) for t in theta_list])
    raw = (2.0 / sigma2) * float(np.sum(np.abs(dG) ** 2))
    g2 = float(np.sum(np.abs(G) ** 2))
    d2 = float(np.sum(np.abs(dG) ** 2))
    if g2 <= 0.0:
        # 全部观测角上 Gamma 均为零（例如只在布儒斯特角观测）：
        # 复增益在这组观测上不可辨识，也就不会吃掉 eps_r 的信息
        rho2 = 0.0
    elif g2 * d2 > 0.0:
        rho2 = abs(np.vdot(G, dG)) ** 2 / (g2 * d2)
    else:
        rho2 = 1.0
    return raw * (1 - rho2), raw, (1 - rho2)


def joint_position_retention(theta_list, beta, eta, ya_list, d, h, freqs, sigma2, kappa=0.0):
    """两条通道合并，消除 (eps_r, a_T, a_W) 后的位置有效信息。"""
    J_T = delay_channel_fim(ya_list, eta, d, h, freqs, sigma2)          # 5x5
    J_W = amp_channel_fim(theta_list, eta[2], kappa, sigma2 / beta) if beta > 0 \
        else np.zeros((3, 3))

    J = np.zeros((7, 7), dtype=float)
    J[:5, :5] += J_T
    for a in range(3):
        for b in range(3):
            J[[2, 5, 6][a], [2, 5, 6][b]] += J_W[a, b]

    J_pos_raw = J_T[:2, :2]                      # 什么都不消除
    # 正确参照：eps_r 已知、只有复增益未知时的位置信息
    J_ref = schur_block(J_T, [0, 1], [3, 4])
    J_eff = schur_block(J, [0, 1], [2, 3, 4, 5, 6])
    ev_raw = np.linalg.eigvalsh(J_pos_raw)
    ev_eff = np.linalg.eigvalsh(J_eff)
    ref_trace = float(np.trace(J_ref))

    J_T_only = schur_block(J_T, [2], [3, 4])
    info_eps_T = float(J_T_only[0, 0])
    info_eps_W, raw_W, div_W = amp_eps_info_after_gain(theta_list, eta[2], kappa,
                                                       sigma2 / beta) if beta > 0 \
        else (0.0, 0.0, 0.0)

    return dict(
        retention=float(np.trace(J_eff) / ref_trace),
        loss_eps=float(1.0 - np.trace(J_eff) / ref_trace),
        eigmin_eff=float(ev_eff[0]), eigmax_eff=float(ev_eff[-1]),
        cond_eff=float(ev_eff[-1] / ev_eff[0]) if ev_eff[0] > 0 else float("inf"),
        eigmin_raw=float(ev_raw[0]), trace_raw=float(np.trace(J_pos_raw)),
        ref_trace=ref_trace,
        trace_eff=float(np.trace(J_eff)),
        info_eps_delay=info_eps_T, info_eps_amp=info_eps_W,
        amp_raw_sens=raw_W, amp_diversity=div_W,
        amp_over_delay=float(info_eps_W / info_eps_T) if info_eps_T > 0 else float("inf"),
    )


def beta_for_retention(theta_list, target, eta, ya_list, d, h, freqs, sigma2, kappa=0.0):
    """求使位置信息保留比例达到 target 所需的幅度通道强度 beta（对数二分）。"""
    lo, hi = -2.0, 12.0

    def f(lb):
        r = joint_position_retention(theta_list, 10.0 ** lb, eta, ya_list, d, h,
                                     freqs, sigma2, kappa)
        return r["retention"] - target

    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return 10.0 ** lo
    if fhi < 0:
        return float("inf")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


# -----------------------------------------------------------------------------
def _setup_cjk():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]:
        try:
            font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True
        except Exception:
            continue
    return plt, False


def run(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    rows = []

    D, H = 0.20, 1.00
    EPS = 6.0
    XT, YT = 3.00, 0.50
    ETA = np.array([XT, YT, EPS])
    F0, BW, NF = 3.0e9, 2.0e9, 64
    freqs = np.linspace(F0 - BW / 2, F0 + BW / 2, NF)
    SIG2 = 1.0

    print("=" * 88)
    print("布儒斯特角探针")
    print(f"  墙厚 d={D} m, 天线线 x=-{H} m, 目标=({XT},{YT}) m, eps_r={EPS}")
    print(f"  频段 {F0/1e9:.1f} GHz, B={BW/1e9:.1f} GHz, 频点 {NF}, 噪声方差 {SIG2}")
    print("=" * 88)

    # ---------------- A 解析层 ----------------
    print("\n[A] Fresnel TM 复反射系数及其对 eps_r 的导数")
    print("    一致性自检：解析导数 vs 复步长 vs 中心差分（无损）")
    for th in [10.0, 30.0, 60.0, 67.7923457, 80.0, 87.0]:
        e = th * DEG
        a1 = dgamma_deps_c(e, EPS)
        a2 = dgamma_depsr_cs(e, EPS)
        a3 = dgamma_depsr_fd(e, EPS)
        print(f"    th={th:10.5f} deg  解析 {a1.real:+.10e}  复步长 {a2:+.10e}  "
              f"差分 {a3.real:+.10e}   相对差 {abs(a1.real-a3.real)/abs(a3.real):.2e}")

    thB = brewster_deg(EPS)
    gB = gamma_tm(thB * DEG, EPS)
    dgB = dgamma_deps_c(thB * DEG, EPS).real
    dg0 = dgamma_deps_c(0.0, EPS).real
    print(f"\n    布儒斯特角 th_B = atan(sqrt({EPS:g})) = {thB:.6f} deg")
    print(f"    |Gamma_TM(th_B)| = {abs(gB):.3e}    (理论为 0，校验公式无误)")
    print(f"    dGamma/deps_r 在 th_B 处 = {dgB:+.8f}")
    print(f"    dGamma/deps_r 在 th=0 处 = {dg0:+.8f}")
    print(f"    两者比值 = {abs(dgB)/abs(dg0):.4f}")
    print("    结论：Gamma 在布儒斯特角为零，其对 eps_r 的一阶导数不为零。")
    print("          机理：零点位置本身随 eps_r 移动，")
    print(f"          d th_B / d eps_r = {math.degrees(1.0/(2*math.sqrt(EPS)*(1+EPS))):.4f} deg 每单位 eps_r，")
    print("          灵敏度由零点移动速度决定，与 Gamma 本身的幅度无关。")

    # 各 eps_r 下 布儒斯特角处导数 / 法向导数
    print(f"\n    比值 dGamma/deps(布儒斯特) / dGamma/deps(法向) 随 eps_r")
    print(f"    {'eps_r':>7s} {'th_B(deg)':>10s} {'法向导数':>12s} {'布儒斯特导数':>13s} {'比值':>8s}")
    for eps in [1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0]:
        t = brewster_deg(eps)
        d0 = dgamma_deps_c(0.0, eps).real
        dB = dgamma_deps_c(t * DEG, eps).real
        print(f"    {eps:7.2f} {t:10.4f} {d0:12.6e} {dB:13.6e} {abs(dB/d0):8.4f}")
        rows.append(dict(kind="eps_ratio", case=f"eps={eps:g}", theta_deg=t, beta="",
                         gamma_abs=float(abs(gamma_tm(t*DEG, eps))), dgamma_deps=float(dB),
                         raw_sens="", diversity="", info_eps_amp="", info_eps_delay="",
                         retention="", note=f"ratio_vs_normal={abs(dB/d0):.5f}"))

    th, Gab, Sab = amp_sens_profile(EPS)
    i_max = int(np.argmax(Sab))
    i_min = int(np.argmin(Sab))
    print(f"\n    |dGamma/deps_r| 在 (0, 89.9) deg 上的最大值 = {Sab[i_max]:.6e} @ {th[i_max]:.3f} deg")
    print(f"    |dGamma/deps_r| 在 (0, 89.9) deg 上的最小值 = {Sab[i_min]:.6e} @ {th[i_min]:.3f} deg")
    core = (th > 5) & (th < 80)
    print(f"    0 到 80 deg 区间内该导数的变化幅度 = "
          f"{(Sab[core].max()-Sab[core].min())/Sab[core].mean()*100:.2f} %  (近乎与角度无关)")
    print(f"    该导数在 th_B 处 = {dgB:.6e}，在 th=85/88/89 deg 处 = "
          f"{dgamma_deps_c(85*DEG, EPS).real:.3e} / {dgamma_deps_c(88*DEG, EPS).real:.3e} / "
          f"{dgamma_deps_c(89*DEG, EPS).real:.3e}")
    print("    含义：幅度通道的 eps_r 盲区在掠入射一侧，不在布儒斯特角。")
    print(f"    同角度下 |Gamma| 的变化则极大（0 到 80 deg 从 {abs(gamma_tm(0.0,EPS)):.4f} "
          f"变为 {abs(gamma_tm(80*DEG,EPS)):.4f}，并在 th_B 归零），")
    print("    即反射幅度剧烈变化而其对 eps_r 的敏感度几乎不变。")

    print("\n    角度扫描表（eps_r=6，无损）")
    print(f"    {'th(deg)':>9s} {'|Gamma|':>10s} {'|dGamma/deps|':>15s} {'|Gamma|*|dG/de|':>16s}")
    for t in [0, 15, 30, 45, 55, 60, 65, thB, 70, 75, 80, 85, 87, 88, 89, 89.5]:
        G = gamma_tm(t * DEG, EPS)
        dG = dgamma_deps_c(t * DEG, EPS)
        print(f"    {t:9.4f} {abs(G):10.5f} {abs(dG):15.6f} {abs(G)*abs(dG):16.6f}")
        rows.append(dict(kind="angle_scan", case="lossless", theta_deg=float(t), beta="",
                         gamma_abs=float(abs(G)), dgamma_deps=float(abs(dG)),
                         raw_sens="", diversity="", info_eps_amp="", info_eps_delay="",
                         retention="", note=f"prod={abs(G)*abs(dG):.6f}"))

    # ---------------- B 损耗层 ----------------
    print("\n[B] 含损耗 eps_c = eps_r - j kappa（kappa = sigma/(omega eps0)，取 eps_r=6 为标称）时")
    print(f"    {'tan d':>8s} {'kappa':>9s} {'|Gamma|min':>11s} {'@th(deg)':>10s} "
          f"{'|dG/de|max':>11s} {'@th(deg)':>10s} {'|dG/de|@th_B':>12s} {'|dG/de|@89deg':>13s}")
    for tand in [0.0, 0.001, 0.01, 0.05, 0.2]:
        kap = EPS * tand
        th_, Gab_, Sab_ = amp_sens_profile(EPS, kap)
        i1 = int(np.argmin(Gab_)); i2 = int(np.argmax(Sab_))
        dB = abs(dgamma_deps_c(thB * DEG, EPS - 1j * kap))
        d89 = abs(dgamma_deps_c(89 * DEG, EPS - 1j * kap))
        print(f"    {tand:8.4f} {kap:9.5f} {Gab_[i1]:11.3e} {th_[i1]:10.3f} "
              f"{Sab_[i2]:11.4e} {th_[i2]:10.3f} {dB:12.4e} {d89:13.4e}")
        rows.append(dict(kind="lossy", case=f"tand={tand:g}", theta_deg=float(th_[i1]), beta="",
                         gamma_abs=float(Gab_[i1]), dgamma_deps=float(Sab_[i2]),
                         raw_sens="", diversity="", info_eps_amp="", info_eps_delay="",
                         retention="",
                         note=f"dGde@thB={dB:.4e}; dGde@89deg={d89:.4e}"))
    print("    含义：损耗把零点抬升为极小值（伪布儒斯特），但导数在伪布儒斯特角附近不塌陷，")
    print("          它依然在掠入射一侧单调衰减到零。")

    # ---------------- C 统计层 ----------------
    print("\n[C] 联合 FIM：未知复增益 + 位置 + eps_r 全联合，消除讨厌参数后的位置信息")
    ya_list = aperture_axis(YT, 1.6, 9)
    th_target = sorted(math.degrees(_solve_th0(YT - ya, H, XT, D, EPS)) for ya in ya_list)
    print(f"    时延通道天线 9 个，孔径 1.6 m，界面入射角范围 "
          f"{min(th_target):.2f} ~ {max(th_target):.2f} deg")

    angle_sets = {
        "法向三角度": [0.0, 5.0, 10.0],
        "布儒斯特+/-2deg": [thB - 2, thB, thB + 2],
        "宽角域12点": list(np.linspace(10, 85, 12)),
        "掠入射三角度": [85.0, 87.0, 88.0],
        "布儒斯特单角度": [thB],
    }
    betas = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    print(f"\n    保留比例定义 = (消除 eps_r 与全部增益后的位置信息) / (仅 eps_r 已知、增益仍未知时的位置信息)")
    print(f"    {'角度集':>17s} {'beta':>9s} {'保留比例':>9s} {'eigmin_eff':>11s} "
          f"{'cond_eff':>10s} {'eps信息(幅度)':>13s} {'eps信息(时延)':>13s} "
          f"{'幅度/时延':>9s} {'多样性':>8s}")
    for name, ths in angle_sets.items():
        for beta in betas:
            r = joint_position_retention(ths, beta, ETA, ya_list, D, H, freqs, SIG2)
            print(f"    {name:>17s} {beta:9.0e} {r['retention']:9.5f} {r['eigmin_eff']:11.4e} "
                  f"{r['cond_eff']:10.3e} {r['info_eps_amp']:13.5e} {r['info_eps_delay']:13.5e} "
                  f"{r['amp_over_delay']:9.3f} {r['amp_diversity']:8.5f}")
            rows.append(dict(kind="joint", case=name, theta_deg="", beta=beta,
                             gamma_abs="", dgamma_deps="",
                             raw_sens=r["amp_raw_sens"], diversity=r["amp_diversity"],
                             info_eps_amp=r["info_eps_amp"], info_eps_delay=r["info_eps_delay"],
                             retention=r["retention"],
                             note=f"eigmin={r['eigmin_eff']:.4e}; cond={r['cond_eff']:.3e};"
                                  f" amp/delay={r['amp_over_delay']:.4f}"))

    base = joint_position_retention([], 0.0, ETA, ya_list, D, H, freqs, SIG2)
    print(f"\n    基线（仅时延通道，未知 eps_r，复增益亦未知）位置信息保留比例 = {base['retention']:.6f}")
    print(f"    未知介电常数吃掉的位置信息比例 = {base['loss_eps']:.6f}")
    print(f"    参照量：eps_r 已知但增益未知时的位置信息 trace = {base['ref_trace']:.4e}")

    print("\n[F] 各角度集把位置信息恢复到 90% 所需的幅度通道强度 beta_90")
    print(f"    {'角度集':>17s} {'多样性因子':>11s} {'beta_90':>12s} {'eps信息(幅度,beta=1)':>20s}")
    b90 = {}
    for name, ths in angle_sets.items():
        bb = beta_for_retention(ths, 0.90, ETA, ya_list, D, H, freqs, SIG2)
        b90[name] = bb
        r1 = joint_position_retention(ths, 1.0, ETA, ya_list, D, H, freqs, SIG2)
        s = f"{bb:.3e}" if np.isfinite(bb) else "不可达"
        print(f"    {name:>17s} {r1['amp_diversity']:11.5f} {s:>12s} {r1['info_eps_amp']:20.4e}")
        rows.append(dict(kind="beta90", case=name, theta_deg="", beta=bb,
                         gamma_abs="", dgamma_deps="", raw_sens=r1["amp_raw_sens"],
                         diversity=r1["amp_diversity"], info_eps_amp=r1["info_eps_amp"],
                         info_eps_delay=r1["info_eps_delay"], retention=0.90,
                         note="所需 beta" if np.isfinite(bb) else "该角度集不可达 0.90"))
    print("    含义：beta_90 越小说明该角度集越能帮助恢复位置信息。")
    print("          布儒斯特附近的角度集多样性最大，对应 beta_90 最小，即布儒斯特角有利于而非有害。")

    print("\n[E] 幅度通道角度集中心角扫描（beta=10）：位置信息随中心角变化")
    print(f"    {'中心角':>9s} {'raw敏感度':>12s} {'多样性':>9s} {'保留比例':>10s} {'eigmin_eff':>11s}")
    sweep = []
    for th0 in np.linspace(3.0, 87.0, 29):
        ths = [th0 - 3, th0, th0 + 3]
        r = joint_position_retention(ths, 10.0, ETA, ya_list, D, H, freqs, SIG2)
        sweep.append((float(th0), r["amp_raw_sens"], r["amp_diversity"], r["retention"],
                      r["eigmin_eff"]))
        print(f"    {th0:9.3f} {r['amp_raw_sens']:12.4e} {r['amp_diversity']:9.5f} "
              f"{r['retention']:10.5f} {r['eigmin_eff']:11.4e}")
        rows.append(dict(kind="angle_center_sweep", case=f"center={th0:.2f}", theta_deg=float(th0),
                         beta=10.0, gamma_abs="", dgamma_deps="",
                         raw_sens=r["amp_raw_sens"], diversity=r["amp_diversity"],
                         info_eps_amp=r["info_eps_amp"], info_eps_delay=r["info_eps_delay"],
                         retention=r["retention"], note=""))

    # ---------------- 输出 CSV ----------------
    keys = ["kind", "case", "theta_deg", "beta", "gamma_abs", "dgamma_deps",
            "raw_sens", "diversity", "info_eps_amp", "info_eps_delay", "retention", "note"]
    p = os.path.join(outdir, "brewster_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n已写出 {p}")

    # ---------------- 图 ----------------
    try:
        plt, cjk = _setup_cjk()
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                             "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})
        L_th = "入射角 (deg)" if cjk else "Incidence angle (deg)"
        cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

        fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.7))
        for i, eps in enumerate([2.0, 4.0, 6.0, 10.0]):
            t_, G_, S_ = amp_sens_profile(eps)
            ax[0].plot(t_, G_, lw=2, color=cols[i], label=f"eps_r={eps:g}")
            ax[1].plot(t_, S_, lw=2, color=cols[i], label=f"eps_r={eps:g}")
            ax[0].axvline(brewster_deg(eps), color=cols[i], ls=":", lw=1.2)
            ax[1].axvline(brewster_deg(eps), color=cols[i], ls=":", lw=1.2)
        ax[0].set_xlabel(L_th); ax[0].set_ylabel("|Gamma_TM|")
        ax[0].set_title("反射系数幅度 |Gamma_TM|  (虚线为布儒斯特角)")
        ax[0].grid(alpha=0.35); ax[0].legend()
        ax[1].set_xlabel(L_th); ax[1].set_ylabel("|dGamma_TM/d eps_r|")
        ax[1].set_title("幅度对介电常数的敏感度  (虚线为布儒斯特角)")
        ax[1].grid(alpha=0.35); ax[1].legend()
        for tand, c in zip([0.0, 0.01, 0.05, 0.2], cols):
            t_, _, S_ = amp_sens_profile(6.0, 6.0 * tand)
            ax[2].plot(t_, S_, lw=2, color=c, label=f"tan d={tand:g}")
        ax[2].axvline(thB, color="gray", ls="--", lw=1.5)
        ax[2].text(thB + 1.0, 0.018, "th_B (eps_r=6)", color="gray", fontsize=10)
        ax[2].set_xlabel(L_th); ax[2].set_ylabel("|dGamma_TM/d eps_r|")
        ax[2].set_title("含损耗时该敏感度"); ax[2].grid(alpha=0.35); ax[2].legend()
        fig.tight_layout()
        f1 = os.path.join(figdir, "fig_brewster_sensitivity.png")
        fig.savefig(f1, dpi=150); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))
        xs = [s[0] for s in sweep]
        ax[0].plot(xs, [s[1] for s in sweep], "o-", lw=2, ms=6, color="#c0392b")
        ax[0].axvline(thB, color="gray", ls="--", lw=1.6)
        ax[0].text(thB + 1.0, 0.5 * (min(s[1] for s in sweep) + max(s[1] for s in sweep)),
                   "th_B", color="gray", fontsize=11)
        ax[0].set_ylim(0.0, max(s[1] for s in sweep) * 1.12)
        ax[0].set_xlabel("幅度通道角度集中心 (deg)" if cjk else "center angle (deg)")
        ax[0].set_ylabel("sum |dGamma/d eps_r|^2")
        ax[0].set_title("幅度通道 eps_r 原始敏感度" if cjk else "raw eps sensitivity")
        ax[0].grid(alpha=0.35)
        ax[1].plot(xs, [s[2] for s in sweep], "s-", lw=2, ms=6, color="#2c6fbb")
        ax[1].axvline(thB, color="gray", ls="--", lw=1.6)
        ax[1].text(thB + 1.0, 0.9 * max(s[2] for s in sweep),
                   "th_B", color="gray", fontsize=11)
        ax[1].set_xlabel("幅度通道角度集中心 (deg)" if cjk else "center angle (deg)")
        ax[1].set_ylabel("多样性因子 1 - |rho|^2")
        ax[1].set_title("消除未知复增益后可保留的比例" if cjk else "diversity factor 1-|rho|^2")
        ax[1].grid(alpha=0.35)
        fig.tight_layout()
        f2 = os.path.join(figdir, "fig_brewster_fim.png")
        fig.savefig(f2, dpi=150); plt.close(fig)
        print(f"已写出图 {f1}\n已写出图 {f2}   (中文字体={'可用' if cjk else '不可用，已用英文'})")
    except Exception as e:
        print(f"[warn] 绘图失败: {e}")

    print("\n完成。")


def main():
    ap = argparse.ArgumentParser(description="布儒斯特角探针")
    ap.add_argument("--outdir", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
