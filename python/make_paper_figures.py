#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_paper_figures.py -- 生成系统级贡献论文的图 1 到图 6（中英两版）

图清单与用途见 `docs/literature/07_系统级贡献定位与论文骨架.md` 第七节，
生成说明与自查记录见 `docs/notes/论文图表清单与生成说明.md`。

数据来源
--------
图 1  几何示意（左侧手绘）＋代表链路的到达方向扇形（WiPTR/val_grid 第 0 层实取）
图 2  results/c2full_*.csv（约 461 万条链路）
图 3  同上（含按锚点数分层）
图 4  results/c2full_*.csv（跨 split）＋ results/c2bw_scan.csv（跨带宽）
图 5  results/c2truth_all.csv（5303 条三路真值）
图 6  results/c2full_*.csv（判据通过条件下的实用后果）

输出
----
results/figures/en/figN_xxx.png 与 .pdf（矢量，pdf.fonttype=42）
results/figures/zh/figN_xxx.png

用法
----
python make_paper_figures.py            # 全部图、全部语言
python make_paper_figures.py --only 2 6 --lang en
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_criterion_audit import fim, fim_metrics, path_directions, path_power  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RES / "figures"

FULL_COLS = ["n_path", "los_frac", "n_anchor", "rx_z", "mean_abs_uz",
             "d_eff", "kappa", "rank", "share_z", "var_z", "lam_max"]

SPLIT_LABEL = {
    "WiPTR_train": "WiPTR train", "WiPTR_train_grid": "WiPTR train_grid",
    "WiPTR_val": "WiPTR val", "WiPTR_val_grid": "WiPTR val_grid",
    "WiPTR_val_ood": "WiPTR val_ood", "WiPTR_test": "WiPTR test",
    "Wi3R_grid": "Wi3R grid", "Wi3R_main_5tx": "Wi3R main_5tx",
    "Wi3R_outofdistribution_fourrooms": "Wi3R ood fourrooms",
    "Wi3R_outofdistribution_scaled": "Wi3R ood scaled",
    "Wi3R_outofdistribution_squared": "Wi3R ood squared",
}
COPLANAR = {k: (k.startswith("WiPTR")) for k in SPLIT_LABEL}

# ---------------------------------------------------------------- 标签
L = {
 "en": dict(
   mean_abs_uz=r"mean $|u_z|$ (arrival direction verticality)",
   share_z=r"$s_z$ (vertical variance share)",
   deff=r"$d_{\mathrm{eff}}$ (effective dimension)",
   kappa=r"$\kappa$ (condition number)",
   fig2_title="Rank and condition number pass while effective dimension fails",
   fig2a_title="(a) Criterion-pass region vs. information structure",
   fig2b_title="(b) Vertical variance share: all links vs. criterion-passing links",
   fig3_title="Vertical degeneration is geometric, not resolution limited",
   fig3a_title="(a) Vertical share vs. arrival-direction verticality",
   fig3b_title="(b) Vertical share vs. number of anchors",
   fig4_title="Failure rate across datasets and bandwidths",
   fig4a_title="(a) Criterion-passing but vertically degenerate links",
   fig4b_title="(b) Rank collapse under resolution limits vs. bandwidth",
   fig5_title="Independent truth: the criterion CRB is accurate",
   fig5a_title="(a) Predicted vs. achieved vertical standard deviation",
   fig5b_title="(b) Multi-start ambiguity vs. number of anchors",
   fig6_title="Practical consequence: vertical degeneracy survives every pass criterion",
   fig6a_title="(a) Links with $s_z>0.5$",
   fig6b_title="(b) Vertical vs. best-axis std. dev. ratio",
   fig7_title="End-to-end check: direction-resolved diagnostics track the vertical error",
   fig7a_title="(a) Vertical error vs. vertical variance share",
   fig7b_title="(b) Screening skill (AUROC) of each criterion, by dataset",
   frac_total="share of all links",
   pass_vs_all="criterion passes\n(rank 3 & $\\kappa<100$)",
   all_links="all links",
   med="median",
   threshold="criterion threshold",
   ambiguous="near-mirror ambiguity",
   one_to_one="1:1",
 ),
 "zh": dict(
   mean_abs_uz=r"平均 $|u_z|$（到达方向的垂直性）",
   share_z=r"$s_z$（垂直方向方差份额）",
   deff=r"$d_{\mathrm{eff}}$（有效维数）",
   kappa=r"$\kappa$（条件数）",
   fig2_title="秩与条件数判通过，而有效维数已失效",
   fig2a_title="(a) 判据通过区间与真实信息结构",
   fig2b_title="(b) 垂直方差份额：全体链路与判据通过链路对比",
   fig3_title="垂直退化是几何效应，不是分辨力限制",
   fig3a_title="(a) 垂直份额与到达方向垂直性的关系",
   fig3b_title="(b) 垂直份额与锚点数的关系",
   fig4_title="跨数据集与跨带宽的失效比例",
   fig4a_title="(a) 判据通过但垂直退化的链路占比",
   fig4b_title="(b) 分辨率受限下的秩跌落与带宽的关系",
   fig5_title="独立真值：判据 CRB 是准确的",
   fig5a_title="(a) 判据预测的垂直标准差与实测垂直标准差",
   fig5b_title="(b) 多起点歧义比例与锚点数的关系",
   fig6_title="实用后果：垂直退化在每一档判据通过条件下都存留",
   fig6a_title="(a) 垂直方差份额大于 0.5 的链路占比",
   fig6b_title="(b) 垂直方向与最好轴的标准差倍数",
   fig7_title="端到端检验：方向分辨诊断量追踪了实际垂直误差",
   fig7a_title="(a) 垂直误差与垂直方差份额的关系",
   fig7b_title="(b) 各判据量作为筛查规则的判别力（AUROC），按数据集分组",
   frac_total="占全体链路比例",
   pass_vs_all="判据通过\n（秩 3 且 $\\kappa<100$）",
   all_links="全体链路",
   med="中位",
   threshold="判据阈值",
   ambiguous="近镜像歧义",
   one_to_one="1:1",
 ),
}


def style(lang: str):
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 12, "axes.titlesize": 14.5, "axes.labelsize": 13,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "legend.fontsize": 11,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.linewidth": 0.9, "lines.linewidth": 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    if lang == "zh":
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Microsoft YaHei", "SimHei",
                                                 "DejaVu Sans"]})
    else:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "DejaVu Serif"],
                             "mathtext.fontset": "stix"})


def save(fig, name: str, lang: str, layouts: str = "constrained"):
    d = FIGDIR / lang
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") if lang == "en" else ("png",):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  写出 {d.name}/{name}.[{'png,pdf' if lang == 'en' else 'png'}]")


# ---------------------------------------------------------------- 数据
def load_full() -> dict:
    data = {}
    for p in sorted(RES.glob("c2full_*.csv")):
        tag = p.stem[len("c2full_"):]
        df = pd.read_csv(p, usecols=FULL_COLS, dtype="float32")
        data[tag] = {c: np.asarray(df[c].to_numpy()) for c in FULL_COLS}
        del df
    return data


def concat(data: dict) -> dict:
    return {c: np.concatenate([data[k][c] for k in sorted(data)]) for c in FULL_COLS}


def load_truth() -> dict:
    df = pd.read_csv(RES / "c2truth_all.csv")
    return {c: df[c].to_numpy() for c in df.columns}


def load_bw() -> pd.DataFrame:
    return pd.read_csv(RES / "c2bw_scan.csv")


def scan_stats(F: dict) -> dict:
    rk, kap, de = F["rank"], F["kappa"], F["d_eff"]
    sz, uz, na = F["share_z"], F["mean_abs_uz"], F["n_anchor"]
    vert = np.sqrt(np.clip(F["var_z"] * F["lam_max"], 0, None))
    m = (rk == 3) & (kap < 100)
    st = dict(
        n=sz.size,
        rank3=float((rk == 3).mean()),
        kappa_med=float(np.median(kap)), kappa_p90=float(np.percentile(kap, 90)),
        deff_med=float(np.median(de)), deff_lt2=float((de < 2).mean()),
        sharez_med=float(np.median(sz)), sharez_gt05=float((sz > 0.5).mean()),
        sharez_gt13=float((sz > 1 / 3).mean()),
        strict=float(((rk == 3) & (kap < 100) & (sz > 0.5)).mean()),
        corr=float(np.corrcoef(sz, uz)[0, 1]),
        vert_med=float(np.median(vert)), vert_p90=float(np.percentile(vert, 90)),
        n_pass=int(m.sum()), pass_frac=float(m.mean()),
        pass_sharez_gt05=float((sz[m] > 0.5).mean()),
        pass_vert_med=float(np.median(vert[m])),
    )
    for th in (30, 100, 300):
        mm = (rk == 3) & (kap < th)
        st[f"k{th}_frac"] = float(mm.mean())
        st[f"k{th}_sharez_gt05"] = float((sz[mm] > 0.5).mean())
        st[f"k{th}_vert_med"] = float(np.median(vert[mm]))
    return st


# ---------------------------------------------------------------- 图 1
def fig1(lang: str):
    t = L[lang]
    Zh, Za = 0.20, 2.30                    # 接收机低层高度、锚点平面高度
    tx = [(1.0, 2.30), (2.2, 2.30), (3.4, 2.30), (5.0, 2.30)]
    scr_x, scr_h = 3.0, 2.6                # 遮挡屏（绕射边）
    wall_x = 5.9
    rx1, rx2 = np.array([4.5, Zh]), np.array([2.0, Za])
    TxA = np.array(tx[0]); TxB = np.array(tx[3])
    E = np.array([scr_x, scr_h])
    Rth = np.array([wall_x, 2.30])          # 墙面高度参考点

    # 反射：虚拟锚点 = 发射机关于墙面的镜像
    VA1 = np.array([2 * wall_x - TxB[0], TxB[1]])
    u1 = (rx1 - VA1) / np.linalg.norm(rx1 - VA1)
    P1 = VA1 + u1 * ((wall_x - VA1[0]) / u1[0])
    # 绕射：虚拟锚点 = 沿到达方向反推整条路径长度
    L1, L2 = np.linalg.norm(E - TxA), np.linalg.norm(rx1 - E)
    u2 = (rx1 - E) / L2
    VA2 = E - L1 * u2

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.6, 5.4),
                                 gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.35", lw=2.2, zorder=1)
    ax.axhline(Za, color="tab:blue", lw=1.4, ls="--", alpha=0.85, zorder=1)
    ax.text(0.02, 0.955, "" if lang == "en" else "", transform=ax.transAxes)
    ax.add_patch(Rectangle((scr_x - 0.07, 0), 0.14, scr_h, color="0.55",
                           zorder=2))
    ax.plot(*E, marker="o", ms=8, color="0.15", zorder=5)
    ax.add_patch(Rectangle((wall_x, 0), 0.14, 3.05, color="0.75", zorder=2))
    for (x, z) in tx:
        ax.plot(x, z, marker="^", ms=11, color="tab:blue", zorder=4)
    ax.plot(*rx1, marker="s", ms=10, color="tab:red", zorder=4)
    ax.plot(*rx2, marker="s", ms=10, color="tab:red", zorder=4)

    # 视距（TxA 到 rx2 同类高度）
    ax.plot([TxA[0], rx2[0]], [TxA[1], rx2[1]], color="0.45", lw=1.4, ls=":",
            zorder=3)
    # 反射路径与镜像虚拟锚点
    ax.plot([TxB[0], P1[0], rx1[0]], [TxB[1], P1[1], rx1[1]], color="tab:green",
            lw=1.9, zorder=3)
    ax.plot([VA1[0], rx1[0]], [VA1[1], rx1[1]], color="tab:green", lw=1.1,
            ls="--", alpha=0.75, zorder=3)
    ax.plot(*VA1, marker="*", ms=17, mfc="none", mec="tab:green", mew=1.6,
            zorder=5)
    ax.text(VA1[0] - 0.02, VA1[1] + 0.26, "$v_1$", color="tab:green",
            fontsize=13, clip_on=True)
    # 绕射路径与等效虚拟锚点
    ax.plot([TxA[0], E[0], rx1[0]], [TxA[1], E[1], rx1[1]], color="tab:purple",
            lw=1.9, zorder=3)
    ax.plot([VA2[0], E[0]], [VA2[1], E[1]], color="tab:purple", lw=1.1, ls="--",
            alpha=0.75, zorder=3)
    ax.plot(*VA2, marker="*", ms=17, mfc="none", mec="tab:purple", mew=1.6,
            zorder=5)
    ax.text(VA2[0] + 0.10, VA2[1] + 0.02, "$v_2$", color="tab:purple",
            fontsize=13, clip_on=True)

    # 垂直杠杆臂标注（Δz 为接收机与锚点所在高度面的垂向间距）
    ax.annotate("", xy=(0.35, Za), xytext=(0.35, Zh),
                arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.4))
    # 水平参考线，用于对比虚拟锚点到接收机的方向与水平面的夹角
    ax.plot([rx1[0] - 1.15, rx1[0] + 0.18], [rx1[1], rx1[1]], color="0.55",
            lw=1.0, ls=":", zorder=3)

    labs = (dict(acc="anchors (one height plane)",
                 rx_lo="receiver, $z$ = 0.20 m",
                 rx_hi="receiver level\nwith the anchors",
                 lev="$\\Delta z$ = 2.10 m", lev0="$\\Delta z$ = 0",
                 ed="diffraction edge", wl="wall", los="line of sight",
                 refl="reflected path", diff="diffracted path",
                 va1="$v_1$: virtual anchor (reflection)",
                 va2="$v_2$: virtual anchor (diffraction)",
                 note="smaller $\\Delta z$:\nflatter arrival\ndirection")
            if lang == "en" else
            dict(acc="锚点（同一高度）",
                 rx_lo="接收机，$z$ = 0.20 m", rx_hi="接收机与锚点同高",
                 lev="$\\Delta z$ = 2.10 m", lev0="$\\Delta z$ = 0",
                 ed="绕射边", wl="墙", los="视距",
                 refl="反射路径", diff="绕射路径",
                 va1="$v_1$：虚拟锚点（反射）",
                 va2="$v_2$：虚拟锚点（绕射）",
                 note="垂向间距 $\\Delta z$ 越小\n方向越接近水平面"))
    ax.plot([], [], color="0.45", lw=1.4, ls=":", label=labs["los"])
    ax.plot([], [], color="tab:green", lw=1.9, label=labs["refl"])
    ax.plot([], [], color="tab:purple", lw=1.9, label=labs["diff"])
    ax.plot([], [], marker="*", ms=15, mfc="none", mec="0.25", ls="none",
            label=labs["va1"])
    ax.plot([], [], marker="*", ms=15, mfc="none", mec="0.25", ls="none",
            label=labs["va2"])
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10.5,
              handlelength=1.9)
    ax.text(-0.50, Za + 0.16, labs["acc"], color="tab:blue", fontsize=11.5,
            va="bottom")
    ax.annotate(labs["rx_lo"], xy=rx1, xytext=(rx1[0] - 0.55, rx1[1] - 0.52),
                color="tab:red", fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
    ax.annotate(labs["rx_hi"], xy=rx2, xytext=(rx2[0] - 1.35, rx2[1] + 0.72),
                color="tab:red", fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
    ax.text(0.47, (Za + Zh) / 2, labs["lev"], color="tab:red", fontsize=11.5,
            va="center")
    ax.text(0.80, 0.11, labs["note"], color="0.15", fontsize=10.5,
            va="bottom", ha="left",
            bbox=dict(fc="white", ec="0.7", alpha=0.92,
                      boxstyle="round,pad=0.30"))
    ax.annotate(labs["lev0"], xy=(2.70, Za + 0.02), xytext=(2.55, Za + 0.42),
                color="0.2", fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color="0.2", lw=1.0))
    ax.text(scr_x + 0.18, scr_h / 2, labs["ed"], color="0.2", fontsize=11.5,
            rotation=90, va="center")
    ax.text(wall_x + 0.24, 0.42, labs["wl"], color="0.35", fontsize=11.5,
            rotation=90, va="center")
    ax.set_xlim(-0.55, 7.15)
    ax.set_ylim(-0.75, 5.35)
    ax.set_xlabel("horizontal position (m)" if lang == "en" else "水平位置（m）")
    ax.set_ylabel("height $z$ (m)" if lang == "en" else "高度 $z$（m）")
    ax.set_title(t["fig1a_title"] if "fig1a_title" in t else
                 ("(a) Two mechanisms, one virtual-anchor constraint"
                  if lang == "en" else "(a) 两种机制，同一类虚拟锚点约束"))

    # panel (b)：代表链路的到达方向扇形
    info = {s.tag: s for s in wio.discover_splits(ROOT)}["WiPTR/val_grid"]
    with h5py.File(info.path, "r") as f:
        fid = [i for i, _ in wio.list_floors(f)][0]
        fc = wio.read_floor(f, fid)
        paths, ch_of = wio.read_floor_paths(f, fid)
    dirs = path_directions(paths)
    w = path_power(paths)
    _, uid = np.unique(fc.rx_xyz, axis=0, return_inverse=True)
    g = uid[ch_of]
    best = None
    for gi in np.unique(g):
        sel = g == gi
        if sel.sum() < 15:
            continue
        ww = w[sel] / w[sel].max()
        m = fim_metrics(fim(dirs[sel], ww))
        score = abs(m["share_z"] - 0.90)          # 取接近退化典型值的一条
        if best is None or score < best[0]:
            best = (score, dirs[sel], ww, m)
    sz_r, dsel, wsel, m = best
    h = dsel[:, :2]
    hhat = h.mean(axis=0)
    hhat /= np.linalg.norm(hhat)
    r = h @ hhat
    z = dsel[:, 2]
    bx.axhline(0.0, color="0.35", lw=1.2, ls="-")
    bx.axvline(0.0, color="0.35", lw=0.9, ls=":")
    amp = 1.0 + 0.5 * (wsel / wsel.max())
    for i in range(r.size):
        bx.annotate("", xy=(r[i] * amp[i], z[i] * amp[i]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="tab:blue",
                                    lw=1.0, alpha=0.8))
    bx.text(0.02, 0.03,
            (("mean $|u_z|$ = %.3f\narrow length $\\propto$ path power"
              % np.abs(z).mean()) if lang == "en" else
             ("平均 $|u_z|$ = %.3f\n箭头长度正比于路径功率" % np.abs(z).mean())),
            transform=bx.transAxes, fontsize=11.5, va="bottom",
            bbox=dict(fc="white", ec="0.7", alpha=0.92,
                      boxstyle="round,pad=0.35"))
    txt = (("$\\lambda_{max}$:%.2f  $\\lambda_{mid}$:%.3f  $\\lambda_{min}$:%.4f\n"
            "$d_{eff}$ = %.2f   $\\kappa$ = %.0f   $s_z$ = %.2f")
           % (m["lam"][2], m["lam"][1], m["lam"][0], m["d_eff"], m["kappa"],
              m["share_z"])) if lang == "en" else \
          (("$\\lambda_{max}$:%.2f  $\\lambda_{mid}$:%.3f  $\\lambda_{min}$:%.4f\n"
            "$d_{eff}$ = %.2f   $\\kappa$ = %.0f   $s_z$ = %.2f")
           % (m["lam"][2], m["lam"][1], m["lam"][0], m["d_eff"], m["kappa"],
              m["share_z"]))
    bx.text(0.03, 0.96, txt, transform=bx.transAxes, va="top", fontsize=11.5,
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.4"))
    bx.set_xlim(-1.25, 1.25)
    bx.set_ylim(-0.45, 0.45)
    bx.set_xlabel(("horizontal component of arrival direction"
                   if lang == "en" else "到达方向的水平分量"))
    bx.set_ylabel("vertical component $u_z$" if lang == "en"
                  else "$u_z$（垂直分量）")
    bx.set_title("(b) Arrival directions of one link (WiPTR/val_grid)"
                 if lang == "en" else "(b) 某条链路的到达方向（WiPTR/val_grid）")
    ttl = ("Fig. 1  Scenario geometry and virtual anchors"
           if lang == "en" else "图 1  场景几何与虚拟锚点")
    fig.suptitle(ttl, fontsize=15.5, y=1.02)
    print(f"  代表链路 share_z={m['share_z']:.3f} d_eff={m['d_eff']:.3f} "
          f"kappa={m['kappa']:.1f} lam={tuple(round(v, 4) for v in m['lam'])} "
          f"mean|uz|={np.abs(z).mean():.3f}")
    save(fig, "fig1_geometry_virtual_anchor", lang)


# ---------------------------------------------------------------- 图 2
def fig2(lang: str, F: dict, st: dict):
    t = L[lang]
    rk, kap, sz, de = F["rank"], F["kappa"], F["share_z"], F["d_eff"]
    rng = np.random.default_rng(7)
    idx = rng.choice(sz.size, size=min(40000, sz.size), replace=False)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    sc = ax.scatter(np.log10(np.clip(kap[idx], 1e-3, None)), de[idx],
                    c=sz[idx], s=7, cmap="coolwarm", vmin=0.0, vmax=1.0,
                    linewidths=0, alpha=0.75)
    xl = (float(np.log10(max(np.percentile(kap, 0.1), 1e-2))),
          float(np.log10(np.percentile(kap, 99.9))))
    ax.axvline(np.log10(100), color="k", lw=1.6, ls="--")
    ax.axhline(2.0, color="k", lw=1.6, ls="--")
    ax.add_patch(Rectangle((xl[0], 1.0), np.log10(100) - xl[0], 1.0,
                           facecolor="0.5", alpha=0.13, zorder=0, lw=0))
    ax.set_xlim(*xl)
    ax.set_ylim(0.98, 3.02)
    ax.text(np.log10(100) + 0.05, 2.96, "$\\kappa=100$", ha="left", va="top",
            fontsize=11.5, bbox=dict(fc="white", ec="none", alpha=0.75,
                                     boxstyle="round,pad=0.15"))
    ax.text(xl[0] + 0.05, 1.06, "$d_{eff}=2$", fontsize=11.5, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.75,
                      boxstyle="round,pad=0.15"))
    ax.text(np.log10(100) - 0.06, 1.05,
            ("criterion passes, $d_{eff}<2$" if lang == "en"
             else "判据通过但 $d_{eff}<2$"), ha="right", va="bottom",
            fontsize=11, color="0.15",
            bbox=dict(fc="white", ec="none", alpha=0.85,
                      boxstyle="round,pad=0.18"))
    ax.text(0.02, 0.97,
            ("rank 3: %.1f%%   $\\kappa$ median: %.1f\n"
             "$d_{eff}<2$: %.1f%%   strict false pass: %.1f%%")
            % (100 * st["rank3"], st["kappa_med"], 100 * st["deff_lt2"],
               100 * st["strict"]) if lang == "en" else
            ("秩 3 占比 %.1f%%   条件数中位 %.1f\n"
             "$d_{eff}<2$ 占比 %.1f%%   严格假通过 %.1f%%")
            % (100 * st["rank3"], st["kappa_med"], 100 * st["deff_lt2"],
               100 * st["strict"]),
            transform=ax.transAxes, fontsize=11.5, va="top",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.4"))
    ax.set_xlabel(t["kappa"]); ax.set_ylabel(t["deff"])
    ax.set_title(t["fig2a_title"])
    cb = fig.colorbar(sc, ax=ax, pad=0.015); cb.set_label(t["share_z"])

    bins = np.linspace(0, 1, 201)
    m_pass = (rk == 3) & (kap < 100)
    for mask, c, ls, lb in ((np.ones_like(sz, bool), "0.35", "-", t["all_links"]),
                            (m_pass, "tab:red", "-", t["pass_vs_all"])):
        cnt, _ = np.histogram(sz[mask], bins=bins)
        cdf = np.cumsum(cnt) / max(cnt.sum(), 1)
        bx.step(bins[1:], cdf, where="post", color=c, ls=ls, lw=2.0, label=lb)
    bx.axvline(0.5, color="k", lw=1.2, ls=":")
    bx.annotate(("%.1f%% vs %.1f%% above 0.5"
                 % (100 * st["sharez_gt05"], 100 * st["pass_sharez_gt05"]))
                if lang == "en" else
                ("大于 0.5 的比例：全体 %.1f%% 与通过 %.1f%%"
                 % (100 * st["sharez_gt05"], 100 * st["pass_sharez_gt05"])),
                xy=(0.5, 0.55), xytext=(0.06, 0.62), fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.1))
    bx.set_xlabel(t["share_z"]); bx.set_ylabel(("empirical CDF" if lang == "en"
                                                else "经验累积分布"))
    bx.set_xlim(0, 1); bx.set_ylim(0, 1.02)
    bx.legend(loc="lower right", framealpha=0.95)
    bx.set_title(t["fig2b_title"])
    fig.suptitle(("Fig. 2  " if lang == "en" else "图 2  ") + t["fig2_title"],
                 fontsize=15.5, y=1.02)
    print(f"  fig2: 秩3 {100*st['rank3']:.2f}%  kappa中位 {st['kappa_med']:.1f}  "
          f"d_eff<2 {100*st['deff_lt2']:.1f}%  严格假通过 {100*st['strict']:.2f}%  "
          f"通过子集 share_z>0.5 {100*st['pass_sharez_gt05']:.1f}%")
    save(fig, "fig2_criterion_pass_vs_structure", lang)


# ---------------------------------------------------------------- 图 3
def fig3(lang: str, data: dict, F: dict, st: dict):
    t = L[lang]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    uz, sz = F["mean_abs_uz"], F["share_z"]
    h = ax.hist2d(uz, sz, bins=[70, 70], cmap="Blues", norm=LogNorm(vmin=1))
    be = 0.5 * (h[1][1:] + h[1][:-1])
    bm, bq1, bq3 = [], [], []
    for i in range(be.size):
        m = (uz >= h[1][i]) & (uz < h[1][i + 1])
        bm.append(np.median(sz[m]) if m.sum() > 100 else np.nan)
        bq1.append(np.percentile(sz[m], 25) if m.sum() > 100 else np.nan)
        bq3.append(np.percentile(sz[m], 75) if m.sum() > 100 else np.nan)
    bm, bq1, bq3 = np.array(bm), np.array(bq1), np.array(bq3)
    ax.plot(be, bm, color="tab:red", lw=2.4, label=t["med"])
    ax.fill_between(be, bq1, bq3, color="tab:red", alpha=0.18,
                    label=("IQR" if lang == "en" else "四分位间距"))
    fin = np.isfinite(uz) & np.isfinite(sz)
    rr = float(np.corrcoef(uz[fin], sz[fin])[0, 1])
    ax.text(0.97, 0.95, ("Pearson $r$ = %.3f" if lang == "en"
                         else "皮尔逊 $r$ = %.3f") % rr,
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            bbox=dict(fc="white", ec="0.7", alpha=0.9,
                      boxstyle="round,pad=0.35"))
    ax.text(0.96, 0.62,
            ("flat arrival directions:\nvertical share exceeds 0.8"
             if lang == "en" else "到达方向越接近水平：\n垂直份额超过 0.8"),
            transform=ax.transAxes, fontsize=11.5, color="0.15", ha="right",
            va="top",
            bbox=dict(fc="white", ec="0.75", alpha=0.9,
                      boxstyle="round,pad=0.3"))
    st["corr_fig"] = rr
    ax.set_xlabel(t["mean_abs_uz"]); ax.set_ylabel(t["share_z"])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title(t["fig3a_title"])
    cb = fig.colorbar(h[3], ax=ax, pad=0.015)
    cb.set_label(("link count" if lang == "en" else "链路数"))

    na = F["n_anchor"]
    uzall = F["mean_abs_uz"]
    groups = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 99)]
    xs, med, lo, hi, ns, uzm = [], [], [], [], [], []
    for i, (a, b) in enumerate(groups):
        m = (na >= a) & (na <= b)
        if m.sum() < 100:
            continue
        xs.append(i)
        med.append(np.median(sz[m])); lo.append(np.percentile(sz[m], 25))
        hi.append(np.percentile(sz[m], 75)); ns.append(int(m.sum()))
        uzm.append(float(np.median(uzall[m])))
    med, lo, hi = np.array(med), np.array(lo), np.array(hi)
    ax2 = bx
    ax2.bar(xs, med, color="tab:blue", alpha=0.85, width=0.62)
    ax2.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="none", ecolor="0.2",
                 elinewidth=1.4, capsize=4)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(a) if a == b else (">6" if b == 99 else str(a))
                         for a, b in groups][:len(xs)])
    # 柱内标注该档等效虚拟锚点方向的垂直分量中位，取代旧的共面比例
    for i, (n, u) in enumerate(zip(ns, uzm)):
        ax2.text(xs[i], 0.058, "$|u_z|$", ha="center", va="bottom",
                 fontsize=9.5, color="white")
        ax2.text(xs[i], 0.012, "%.2f" % u, ha="center", va="bottom",
                 fontsize=9.5, color="white")
        short = (("%.2fM" % (n / 1e6)) if n >= 1e6 else
                 ("%.0fk" % (n / 1e3)) if n >= 1e3 else str(n))
        ax2.text(xs[i], hi[i] + 0.035, "n=" + short, ha="center", va="bottom",
                 fontsize=9, color="0.25")
    ax2.set_xlabel(("number of anchors" if lang == "en" else "锚点数"))
    ax2.set_ylabel(t["share_z"])
    ax2.set_ylim(0, 1.22)
    ax2.set_xlim(-0.95, len(xs) - 0.35)
    ax2.set_title(t["fig3b_title"])
    ax2.text(0.0, 1.19,
             ("2 to 3 anchors: highest $|u_z|$, lowest $s_z$\n"
              "4 and above: $|u_z|$ falls back, $s_z$ returns"
              if lang == "en" else
              "锚点 2 至 3：$|u_z|$ 最高，$s_z$ 最低\n"
              "锚点 4 及以上：$|u_z|$ 回落，$s_z$ 回升"),
             fontsize=10.5, ha="center", va="top",
             bbox=dict(fc="white", ec="0.7", alpha=0.92,
                       boxstyle="round,pad=0.30"))
    fig.suptitle(("Fig. 3  " if lang == "en" else "图 3  ") + t["fig3_title"],
                 fontsize=15.5, y=1.02)
    print("  fig3: corr=%.3f  锚点分层 share_z 中位 " % st["corr"] +
          ", ".join("%d:%.3f" % (a if a == b else 99, v)
                    for (a, b), v in zip(groups, med)))
    save(fig, "fig3_share_z_vs_geometry", lang)


# ---------------------------------------------------------------- 图 4
def fig4(lang: str, data: dict, bw: pd.DataFrame):
    t = L[lang]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.8, 5.6),
                                 gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    order = sorted(data, key=lambda k: (not COPLANAR[k], SPLIT_LABEL[k]))
    loose, strict = [], []
    for tag in order:
        dd = data[tag]
        rk, kap, sz = dd["rank"], dd["kappa"], dd["share_z"]
        loose.append(100 * np.mean((rk == 3) & (sz > 0.5)))
        strict.append(100 * np.mean((rk == 3) & (kap < 100) & (sz > 0.5)))
    y = np.arange(len(order))
    ax.barh(y - 0.20, loose, height=0.38, color="tab:orange",
            label=("rank 3 & $s_z>0.5$" if lang == "en" else "秩 3 且 $s_z>0.5$"))
    ax.barh(y + 0.20, strict, height=0.38, color="tab:red",
            label=("rank 3 & $\\kappa<100$ & $s_z>0.5$" if lang == "en"
                   else "秩 3 且 $\\kappa<100$ 且 $s_z>0.5$"))
    ax.set_yticks(y)
    ax.set_yticklabels([SPLIT_LABEL[k] for k in order])
    ax.set_ylim(-0.65, len(order) + 0.38)
    ax.axhline(5.5, color="0.3", lw=1.0, ls=":")
    ax.text(99, 5.52, ("anchors coplanar" if lang == "en" else "锚点共面"),
            fontsize=11.5, color="tab:red", ha="right", va="center")
    ax.text(99, len(order) + 0.02,
            ("anchors unequal heights" if lang == "en" else "锚点高度不一"),
            fontsize=11.5, color="tab:blue", ha="right", va="center")
    ax.set_xlabel(("% of links in split" if lang == "en" else "占该 split 链路比例（%）"))
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10.5)
    ax.set_title(t["fig4a_title"])

    bws = sorted(bw["bw_mhz"].unique())
    tags = [s for s in ["WiPTR/train", "WiPTR/val_grid", "WiPTR/val_ood",
                        "Wi3R/main_5tx", "Wi3R/outofdistribution/fourrooms"]
            if s in set(bw["split"])]
    w = 0.8 / len(bws)
    for j, b in enumerate(bws):
        vals = [100 * float(bw[(bw["split"] == s) & (bw["bw_mhz"] == b)][
            "rank_short_both"].iloc[0]) for s in tags]
        bx.bar(np.arange(len(tags)) + (j - (len(bws) - 1) / 2) * w, vals,
               width=w * 0.92,
               label=("%.0f MHz" % b), color=["tab:blue", "tab:green",
                                              "tab:orange"][j % 3])
    bx.set_xticks(np.arange(len(tags)))
    bx.set_xticklabels([SPLIT_LABEL[s.replace("/", "_")] for s in tags],
                       rotation=20, ha="right")
    bx.set_ylabel(("rank 3 but rank-deficient after resolution merge (%)"
                   if lang == "en" else "判据满秩但双重受限后秩不足（%）"))
    bx.legend(title=("bandwidth" if lang == "en" else "带宽"), framealpha=0.95)
    bx.set_title(t["fig4b_title"])
    fig.suptitle(("Fig. 4  " if lang == "en" else "图 4  ") + t["fig4_title"],
                 fontsize=15.5, y=1.02)
    print("  fig4a 宽松/严格: " + ", ".join(
        "%s %.1f/%.1f" % (SPLIT_LABEL[k], a, b) for k, a, b in
        zip(order, loose, strict)))
    save(fig, "fig4_failure_by_split_and_bandwidth", lang)


# ---------------------------------------------------------------- 图 5
def fig5(lang: str, T: dict):
    t = L[lang]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    rz = T["ratio_z"].astype(float)
    crb = T["crb_z"].astype(float)
    bins = T["share_bin"]
    ok = np.isfinite(rz) & np.isfinite(crb) & (rz > 0) & (crb > 0)
    pred = np.sqrt(crb[ok]); ach = np.sqrt(rz[ok] * crb[ok])
    colors = {"low": "tab:blue", "mid": "tab:green", "high": "tab:red"}
    for nm in ("low", "mid", "high"):
        m = bins[ok] == nm
        ax.scatter(pred[m], ach[m], s=9, alpha=0.55, linewidths=0,
                   color=colors[nm],
                   label=(("$s_z$ " + nm) if lang == "en" else
                          ("$s_z$ 低" if nm == "low" else
                           "$s_z$ 中" if nm == "mid" else "$s_z$ 高")))
    lim = (np.nanpercentile(pred, 0.2) * 0.8, np.nanpercentile(pred, 99.8) * 1.25)
    ax.plot(lim, lim, color="k", lw=1.5, ls="--", label=t["one_to_one"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(("predicted vertical std. dev. from CRB (m)" if lang == "en"
                   else "判据 CRB 给出的垂直标准差（m）"))
    ax.set_ylabel(("achieved vertical std. dev., Monte Carlo (m)" if lang == "en"
                   else "蒙特卡洛实测垂直标准差（m）"))
    ax.text(0.03, 0.96,
            (("median ratio $\\sigma_{ach}/\\sigma_{CRB}$\n"
              "$x$ %.3f   $y$ %.3f   $z$ %.3f")
             % (np.median(T["ratio_x"]), np.median(T["ratio_y"]),
                np.median(T["ratio_z"]))) if lang == "en" else
            (("实测/CRB 中位比值\n"
              "$x$ %.3f   $y$ %.3f   $z$ %.3f")
             % (np.median(T["ratio_x"]), np.median(T["ratio_y"]),
                np.median(T["ratio_z"]))),
            transform=ax.transAxes, va="top", fontsize=12,
            bbox=dict(fc="white", ec="0.7", alpha=0.92,
                      boxstyle="round,pad=0.4"))
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10.5)
    ax.set_title(t["fig5a_title"])

    na = T["n_anchor"].astype(int); amb = T["ambig"].astype(float)
    groups = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, ">=4")]
    xs, rate, ns = [], [], []
    for i, (a, b, _) in enumerate(groups):
        m = (na >= a) & (na <= b)
        if m.sum() < 5:
            continue
        xs.append(i); rate.append(100 * amb[m].mean()); ns.append(int(m.sum()))
    bx.bar(xs, rate, color=["tab:red", "tab:blue", "tab:blue", "tab:blue"],
           width=0.6)
    for i, (v, n) in enumerate(zip(rate, ns)):
        bx.text(xs[i], v + 0.12, ("%.2f%%\nn=%d" % (v, n)), ha="center",
                fontsize=11.5, color="0.2")
    bx.set_xticks(xs)
    bx.set_xticklabels([g[2] for g in groups][:len(xs)])
    bx.set_ylim(0, max(rate) * 1.42)
    bx.set_xlabel(("number of anchors" if lang == "en" else "锚点数"))
    bx.set_ylabel(("% of links with a second near-mirror solution" if lang == "en"
                   else "存在第二近镜像解的链路占比（%）"))
    bx.set_title(t["fig5b_title"])
    bx.annotate(("all 5303 links have rank 3" if lang == "en"
                 else "5303 条链路秩全为 3"), xy=(0.02, 0.97),
                xycoords="axes fraction", fontsize=11.5,
                bbox=dict(fc="white", ec="0.7", alpha=0.9,
                          boxstyle="round,pad=0.35"))
    fig.suptitle(("Fig. 5  " if lang == "en" else "图 5  ") + t["fig5_title"],
                 fontsize=15.5, y=1.02)
    print("  fig5: ratio 中位 x %.3f y %.3f z %.3f  歧义按锚点 " %
          (np.median(T["ratio_x"]), np.median(T["ratio_y"]),
           np.median(T["ratio_z"])) +
          ", ".join("%s:%.2f%%" % (g[2], v) for g, v in zip(groups, rate)))
    save(fig, "fig5_independent_truth", lang)


# ---------------------------------------------------------------- 图 6
def fig6(lang: str, F: dict, st: dict):
    t = L[lang]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.2))
    labs = ((r"$\kappa<30$", r"$\kappa<100$", r"$\kappa<300$",
             "no criterion\n(rank 3)") if lang == "en" else
            (r"$\kappa<30$", r"$\kappa<100$", r"$\kappa<300$",
             "不设条件\n（秩 3）"))
    frac = [100 * st["k30_sharez_gt05"], 100 * st["k100_sharez_gt05"],
            100 * st["k300_sharez_gt05"], 100 * st["sharez_gt05"]]
    vert = [st["k30_vert_med"], st["k100_vert_med"], st["k300_vert_med"],
            st["vert_med"]]
    share = [100 * st["k30_frac"], 100 * st["k100_frac"], 100 * st["k300_frac"],
             100 * st["rank3"]]
    x = np.arange(4)
    ax.bar(x, frac, color="tab:red", width=0.62)
    for i, (v, s) in enumerate(zip(frac, share)):
        ax.text(x[i], v + 1.4, "%.1f%%" % v, ha="center", fontsize=12.5)
        ax.text(x[i], 3.0, ("%.0f%% of all" % s) if lang == "en"
                else ("占全体 %.0f%%" % s), ha="center", fontsize=10.5,
                color="white")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel(t["share_z"] + (" > 0.5 (%)" if lang == "en" else " 大于 0.5（%）"))
    ax.set_ylim(0, 72)
    ax.set_title(t["fig6a_title"])

    bx.bar(x, vert, color="tab:purple", width=0.62)
    for i, v in enumerate(vert):
        bx.text(x[i], v + 0.10, "%.2f" % v, ha="center", fontsize=12.5)
    bx.axhline(st["vert_med"], color="0.35", lw=1.3, ls=":")
    bx.set_xticks(x); bx.set_xticklabels(labs)
    bx.set_ylabel(("vertical / best-axis std. dev. (median)" if lang == "en"
                   else "垂直方向与最好轴的标准差倍数（中位）"))
    bx.set_ylim(0, max(vert) * 1.28)
    bx.set_title(t["fig6b_title"])
    fig.suptitle(("Fig. 6  " if lang == "en" else "图 6  ") + t["fig6_title"],
                 fontsize=15.5, y=1.03)
    print("  fig6: share_z>0.5 " + ", ".join("%.1f%%" % v for v in frac) +
          "; 垂直倍数 " + ", ".join("%.2f" % v for v in vert))
    save(fig, "fig6_practical_consequence", lang)


# ---------------------------------------------------------------- 图 7
def fig7(lang: str):
    """端到端对照：判据指标与实际垂直定位误差的关系（results/e2e_links.csv）。"""
    t = L[lang]
    df = pd.read_csv(RES / "e2e_links.csv")
    e = np.abs(df["aoa_sz"].to_numpy(dtype=float))
    sz = df["share_z"].to_numpy(dtype=float)
    kap = df["kappa"].to_numpy(dtype=float)
    de = df["d_eff"].to_numpy(dtype=float)
    uz = df["mean_abs_uz"].to_numpy(dtype=float)
    ok = np.isfinite(e) & np.isfinite(sz)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                 gridspec_kw=dict(width_ratios=[1.15, 1.0]))

    # (a) 垂直误差中位随 share_z 的分位曲线
    bins = np.linspace(0.0, 1.0, 51)
    be = 0.5 * (bins[1:] + bins[:-1])
    med, q1, q3 = [], [], []
    for i in range(be.size):
        m = ok & (sz >= bins[i]) & (sz < bins[i + 1])
        if m.sum() < 50:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
            continue
        med.append(np.median(e[m]))
        q1.append(np.percentile(e[m], 25))
        q3.append(np.percentile(e[m], 75))
    ax.plot(be, med, color="tab:red", lw=2.4, label=t["med"])
    ax.fill_between(be, q1, q3, color="tab:red", alpha=0.18,
                    label=("IQR" if lang == "en" else "四分位间距"))
    ax.axvline(0.5, color="k", lw=1.2, ls=":")
    ax.set_xlabel(t["share_z"])
    ax.set_ylabel(("median vertical position error std. dev. (m)"
                   if lang == "en" else "垂直位置误差标准差中位（m）"))
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_title(t["fig7a_title"])

    # (b) 各判据量作为筛查规则的判别力：AUROC（按数据集分组）
    def auc(score, y):
        m = np.isfinite(score) & np.isfinite(y)
        s, yy = score[m], y[m]
        if yy.sum() == 0 or yy.sum() == yy.size:
            return np.nan
        r = pd.Series(s).rank().to_numpy()
        npos, nneg = yy.sum(), yy.size - yy.sum()
        return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    tau = 0.5
    ylab = (e > tau).astype(int)
    isw = df["split"].str.startswith("WiPTR").to_numpy()
    labs = [r"$\kappa$", r"$d_{\mathrm{eff}}$", r"$s_z$",
            (r"mean $|u_z|$" if lang == "en" else r"平均 $|u_z|$")]
    src = (kap, -de, sz, -uz)
    vw = [auc(s[isw], ylab[isw]) for s in src]
    v3 = [auc(s[~isw], ylab[~isw]) for s in src]
    y = np.arange(4)
    h = 0.36
    bx.barh(y + h / 2, vw, height=h, color="tab:blue", label="WiPTR")
    bx.barh(y - h / 2, v3, height=h, color="tab:orange", label="Wi3R")
    bx.set_yticks(y); bx.set_yticklabels(labs)
    bx.axvline(0.5, color="k", lw=1.0, ls=":")
    for i in range(4):
        bx.text(vw[i] + 0.012, y[i] + h / 2, "%.2f" % vw[i],
                va="center", ha="left", fontsize=11)
        bx.text(v3[i] + 0.012, y[i] - h / 2, "%.2f" % v3[i],
                va="center", ha="left", fontsize=11)
    bx.set_xlim(0.4, 1.02)
    bx.set_xlabel("AUROC" if lang == "en" else "AUROC（垂直误差超限）")
    bx.legend(loc="lower right", framealpha=0.95)
    bx.set_title(t["fig7b_title"])
    fig.suptitle(("Fig. 7  " if lang == "en" else "图 7  ") + t["fig7_title"],
                 fontsize=15.5, y=1.02)
    print("  fig7: AUROC WiPTR " + ", ".join("%s %.3f" % (s, v)
                                            for s, v in zip(("kappa", "deff", "sharez",
                                                             "mean_uz"), vw))
          + " | Wi3R " + ", ".join("%.3f" % v for v in v3))
    save(fig, "fig7_end_to_end", lang)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的图号，缺省全部")
    ap.add_argument("--lang", default="en,zh")
    args = ap.parse_args()
    only = set(int(v) for v in args.only.split(",")) if args.only else {1, 2, 3, 4, 5, 6, 7}
    langs = [s.strip() for s in args.lang.split(",")]

    need_full = bool(only & {2, 3, 4, 6})
    need_truth = 5 in only
    need_bw = 4 in only
    data = load_full() if need_full else {}
    F = concat(data) if need_full else None
    st = scan_stats(F) if need_full else None
    T = load_truth() if need_truth else None
    bw = load_bw() if need_bw else None
    if need_full:
        print("全量 %d 条链路  kappa中位 %.1f  share_z中位 %.4f  "
              "share_z>0.5 %.3f  严格假通过 %.4f  corr %.3f"
              % (F["rank"].size, st["kappa_med"], st["sharez_med"],
                 st["sharez_gt05"], st["strict"], st["corr"]))

    for lang in langs:
        style(lang)
        print(f"--- {lang} ---")
        if 1 in only:
            fig1(lang)
        if 2 in only:
            fig2(lang, F, st)
        if 3 in only:
            fig3(lang, data, F, st)
        if 4 in only:
            fig4(lang, data, bw)
        if 5 in only:
            fig5(lang, T)
        if 6 in only:
            fig6(lang, F, st)
        if 7 in only:
            fig7(lang)


if __name__ == "__main__":
    main()
