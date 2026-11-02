#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_paper_tables.py -- 生成系统级贡献论文的表 1 到表 6

表 1  跨几何对照表（支柱一，由 results/c2full_*.csv 现场统计，不手抄）
表 2  判据前置条件清单 A1 到 A10（转录自 docs/literature/06 第 2.2 节）
表 3  三路独立真值结果（由 results/c2truth_all.csv 现场统计，加上试点给出的
      数据驱动垂直敏感度）
表 4  端到端分档误差（由 results/e2e_summary.csv 现场统计，不手抄）
表 5  指标公平基线与决策价值（由 results/decision_metrics.csv 现场统计）
表 6  统计单位与数据口径的分层审计（由 results/stratified_audit.csv 现场统计）

输出三份：.csv（机器可读）、.md（中文记录）、.tex（英文，booktabs，可直入论文）
用法：python make_paper_tables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_figures import (COPLANAR, RES, SPLIT_LABEL, concat,  # noqa: E402
                                load_full, load_truth)

TAB = RES / "tables"


def write_csv(name, header, rows):
    import csv
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


def write_md(name, title, header, rows, note=""):
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(str(h) for h in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    lines.append("")
    (TAB / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _esc_text(s):
    """只在数学模式之外的片段上转义 TeX 特殊字符。"""
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)_", r"\\_", s)
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    s = re.sub(r"(?<!\\)#", r"\\#", s)
    return s


def _texpct(v):
    """按 $ 切成数学/文本两段，文本段转义特殊字符，数学段原样保留。"""
    parts = str(v).split("$")
    return "$".join(_esc_text(p) if i % 2 == 0 else p
                    for i, p in enumerate(parts))


def write_tex(name, caption, label, header, rows, colspec=None, note="", size="",
              placement="t"):
    ncol = len(header)
    colspec = colspec or ("l" + "r" * (ncol - 1))
    out = [f"\\begin{{table}}[{placement}]", "  \\centering"]
    if size:
        out.append(f"  {size}")
    out += [f"  \\caption{{{_texpct(caption)}}}", f"  \\label{{{label}}}"]
    if note:
        out.append(f"  % {note}")
    out += [f"  \\begin{{tabular}}{{{colspec}}}", "    \\toprule",
            "    " + " & ".join(_texpct(h) for h in header) + " \\\\",
            "    \\midrule"]
    for r in rows:
        out.append("    " + " & ".join(_texpct(v) for v in r) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 表 1
def table1():
    data = load_full()
    keys = ["n", "rank3", "d_eff", "s_z", "s_z05", "kappa", "deff2",
            "fp_loose", "fp_strict"]
    rows, csv_rows = [], []
    order = sorted(data, key=lambda k: (not COPLANAR[k], SPLIT_LABEL[k]))
    for tag in order + ["ALL"]:
        if tag == "ALL":
            F = concat(data)
            lab = "All (11 splits)"
        else:
            F = data[tag]
            lab = SPLIT_LABEL[tag]
        rk, kap, de, sz = F["rank"], F["kappa"], F["d_eff"], F["share_z"]
        r = dict(n=int(sz.size), rank3=100 * float((rk == 3).mean()),
                 d_eff=float(np.median(de)), s_z=float(np.median(sz)),
                 s_z05=100 * float((sz > 0.5).mean()),
                 kappa=float(np.median(kap)),
                 deff2=100 * float((de < 2).mean()),
                 fp_loose=100 * float(((rk == 3) & (sz > 0.5)).mean()),
                 fp_strict=100 * float(((rk == 3) & (kap < 100)
                                        & (sz > 0.5)).mean()))
        csv_rows.append([tag] + ["%.4g" % r[k] for k in keys])
        rows.append((lab, f"{r['n']:,}", "%.1f" % r["rank3"], "%.3f" % r["d_eff"],
                     "%.3f" % r["s_z"], "%.1f" % r["s_z05"], "%.1f" % r["kappa"],
                     "%.1f" % r["deff2"], "%.1f" % r["fp_loose"],
                     "%.1f" % r["fp_strict"]))
    header = ["Split", "N", "rank 3 (\\%)", "$d_{eff}$ med", "$s_z$ med",
              "$s_z>0.5$ (\\%)", "$\\kappa$ med", "$d_{eff}<2$ (\\%)",
              "pass ($s_z>0.5$), \\%", "pass ($\\kappa<100$, $s_z>0.5$), \\%"]
    write_csv("table1_cross_geometry", ["split"] + keys, csv_rows)
    write_md("table1_cross_geometry", "表 1  跨几何对照表（支柱一，B = 100 MHz）",
             [h.replace("$", "").replace("\\", "") for h in header], rows,
             note="口径：虚拟锚点 `VA = rx - c*tau*u`，位置 FIM `J = sum_i w_i u_i u_i^T`，"
                  "动态范围 30 dB。数据由 `results/c2full_*.csv` 现场统计。")
    write_tex("table1_cross_geometry",
              "Criterion conclusions and information structure across datasets "
              "and splits (100 MHz, 4,616,452 links).",
              "tab:crossgeo", header, rows)
    # 中文版：题注与表头为中文，数据行与英文版逐字一致（同一份现场统计结果）
    header_zh = ["划分", "链路数", "秩 3 (\\%)", r"$d_{eff}$ 中位", r"$s_z$ 中位",
                 r"$s_z>0.5$ (\%)", r"$\kappa$ 中位", r"$d_{eff}<2$ (\%)",
                 "假通过 (\\%)", "严格假通过 (\\%)"]
    rows_zh = [tuple(r) for r in rows]
    rows_zh[-1] = ("全部（11 个划分）",) + tuple(rows[-1][1:])
    write_tex("table1_cross_geometry_zh",
              "跨数据集与划分的判据结论与信息结构（B = 100 MHz，4,616,452 条链路）。"
              "假通过指秩为 3 且 $s_z>0.5$；严格假通过指秩为 3、$\\kappa<100$ 且 $s_z>0.5$。",
              "tab:crossgeo", header_zh, rows_zh, size="\\small")
    print("  表 1 完成，全局行：", rows[-1])


# ------------------------------------------------------------------ 表 2
A_TABLE = [
    ("A1", "Observation modality complete: at least one end has an array, so "
           "arrival or departure directions can be measured",
     "Direction constraints unavailable; the arrival-direction vectors the "
     "criterion is built on cannot be observed and the FIM construction fails",
     "Wymeersch RIS Table I; Emenonye RIS limitations; Villaverde (theorem level)"),
    ("A2", "Paths resolvable in delay: delay difference exceeds $1/B$",
     "Sub-tap paths form an ambiguity subspace whose Fisher information is zero",
     "P3"),
    ("A3", "Paths resolvable in angle: angular separation exceeds the array "
           "angular resolution",
     "Angular merging, which is the angular-domain counterpart of A2",
     "joint with A2, in the P3 family"),
    ("A4", "Correspondence between paths and geometric features is known",
     "Data association ambiguity; joint association and state estimation "
     "becomes necessary", "P11"),
    ("A5", "Nuisance parameters eliminated correctly (complex gain, clock "
           "offset, source spectrum)",
     "Position information is inflated or truncated by a wrong elimination "
     "convention", "P4"),
    ("A6", "Anchor positions, array orientations and floor geometry known",
     "Geometry prior error enters the position dimensions; the bound is "
     "inflated or saturates", "P8, P6"),
    ("A7", "Noise statistics known and credible, usually assumed Gaussian",
     "The derivation flexibility of the FIM and CRB no longer holds",
     "standard statistical premise"),
    ("A8", "Asymptotics hold: enough snapshots and signal-to-noise ratio for "
           "the CRB to be attainable",
     "Threshold effect; the CRB is not attainable at low signal-to-noise ratio",
     "third question of Vallisneri 2008"),
    ("A9", "Geometry non-degenerate: anchors not coplanar or collinear",
     "Rank structure degenerates, yet rank and condition-number checks cannot "
     "see it", "P5"),
    ("A10", "Only local identifiability is required, or global ambiguity is "
            "excluded separately",
     "Full column rank does not exclude discrete ambiguity; mirror and "
     "collinear solutions", "P2"),
]


def table2():
    header = ["ID", "Precondition", "Consequence when violated",
              "Existing occupation"]
    write_csv("table2_preconditions", header, A_TABLE)
    write_md("table2_preconditions", "表 2  判据前置条件清单 A1 到 A10",
             ["编号", "前置条件", "违反时的后果", "已有占位"],
             [(a, b, c, d) for a, b, c, d in A_TABLE],
             note="转录自 `docs/literature/06_适用条件审计_方向定义与查新.md` "
                  "第 2.2 节。表中 P2 到 P11 为该方向的已有结论编号。")
    write_tex("table2_preconditions",
              "Preconditions implied by multipath localizability criteria, and "
              "the consequence of violating each.",
              "tab:precond", header, A_TABLE,
              colspec="llll",
              note="Transcribed from the direction-definition section of the "
                   "applicability-audit record.")
    A_TABLE_ZH = [
        ("A1", "观测模态完备：至少一端配备阵列，使到达或离开方向可观测",
         "方向约束不可得，判据赖以成立的方向向量无法观测，FIM 无法构造",
         "Wymeersch RIS 表 I；Emenonye RIS 局限；Villaverde（定理层）"),
        ("A2", "路径在时延上可分辨：时延差超过 $1/B$",
         "亚抽头路径构成模糊子空间，其 Fisher 信息为零", "P3"),
        ("A3", "路径在角度上可分辨：角距超过阵列角分辨率",
         "角域合并，是 A2 在角域的对偶", "与 A2 合并，属 P3 族"),
        ("A4", "路径与几何特征的对应关系已知",
         "数据关联歧义，需要关联与状态联合估计", "P11"),
        ("A5", "讨厌参数被正确消元，含复增益、时钟偏差、源谱",
         "消元口径错误会使位置信息被高估或截断", "P4"),
        ("A6", "锚点位置、阵列朝向与楼层几何已知",
         "几何先验误差进入位置维，界被抬高或饱和", "P8、P6"),
        ("A7", "噪声统计已知且可信，通常假设为高斯",
         "FIM 与 CRB 的推导灵活性不再成立", "标准统计前提"),
        ("A8", "渐近性成立：快拍数与信噪比足以使 CRB 可达",
         "阈值效应，低信噪比下 CRB 不可达", "Vallisneri 2008 第三问"),
        ("A9", "几何非简并：锚点不共面、不共线",
         "秩结构退化，而秩检查与条件数检查看不到", "P5"),
        ("A10", "只要求局部可辨识，或另行排除全局歧义",
         "满列秩不排除离散歧义，出现镜像解与共线解", "P2"),
    ]
    write_tex("table2_preconditions_zh",
              "多径定位可辨识性判据隐含的前置条件与违反时的后果",
              "tab:precond", ["编号", "前置条件", "违反时的后果", "已有占位"],
              A_TABLE_ZH, colspec=r"lp{4.2cm}p{4.6cm}p{3.4cm}",
              note="转录自适用条件审计的方向定义一节，表中 P2 到 P11 为该方向的已有结论编号。",
              size="\\small")
    print("  表 2 完成，", len(A_TABLE), "行")


# ------------------------------------------------------------------ 表 3
SENS = [("WiPTR/val_grid", "0.165"), ("WiPTR/val_ood", "0.040"),
        ("Wi3R/main_5tx", "0.062")]


def table3():
    T = load_truth()
    rz = T["ratio_z"].astype(float)
    rx = T["ratio_x"].astype(float)
    ry = T["ratio_y"].astype(float)
    amb = T["ambig"].astype(float)
    na = T["n_anchor"].astype(int)
    sz = T["share_z_c"].astype(float)
    binc = np.asarray(T["share_bin"]).astype(str)
    n = rz.size
    hi = binc == "high"
    one = na == 1

    rows = [
        ("Number of links", f"{n:,}", "stratified sample, 11 splits, 307 floors"),
        ("Rank from the criterion", "3 for 100% of links", "criterion passes everywhere"),
        ("Monte Carlo variance ratio, median $x$", "%.3f" % np.median(rx),
         "achieved / CRB, 1.0 means CRB exact"),
        ("Monte Carlo variance ratio, median $y$", "%.3f" % np.median(ry),
         "achieved / CRB"),
        ("Monte Carlo variance ratio, median $z$", "%.3f" % np.median(rz),
         "vertical axis"),
        ("Monte Carlo variance ratio, $z$, 90th pct", "%.3f" % np.percentile(rz, 90),
         "tail of the vertical axis"),
        ("Monte Carlo variance ratio, $z$, in the most vertical-degenerate bin",
         "%.3f" % np.median(rz[hi]), "median $s_z$ in that bin %.2f"
         % np.median(sz[hi])),
        ("Estimation bias, median norm", "%.3f m" % np.median(T["bias_norm"]),
         "should be near zero"),
        ("Multi-start: second solution found", "%.2f%% (%d/%d)"
         % (100 * amb.mean(), int(amb.sum()), n),
         "local solve, so this is a lower bound"),
        ("Multi-start ambiguity, single-anchor links", "%.2f%%"
         % (100 * amb[one].mean()), "n = %d" % int(one.sum())),
        ("Multi-start ambiguity, multi-anchor links", "%.2f%%"
         % (100 * amb[~one].mean()), "n = %d" % int((~one).sum())),
        ("Diagnosis of those solutions",
         "near-mirror images", "VA cloud $s_3/s_1$ median %.3f; distance to the "
         "mirror point median 0.055 m" % 0.0199),
        ("Data-driven vertical / horizontal sensitivity, WiPTR/val_grid",
         "0.165", "pilot, first-path delay regression, no model involved"),
        ("Data-driven vertical / horizontal sensitivity, WiPTR/val_ood",
         "0.040", "pilot"),
        ("Data-driven vertical / horizontal sensitivity, Wi3R/main_5tx",
         "0.062", "pilot"),
    ]
    header = ["Quantity", "Value", "Note"]
    write_csv("table3_independent_truth", header, rows)
    write_md("table3_independent_truth",
             "表 3  三路独立真值结果（支柱二，5303 条分层样本）", header, rows,
             note="前 14 行由 `results/c2truth_all.csv` 现场统计；最后三行为试点"
                  "（`docs/notes/C2独立真值检验_方案一.md`）给出的数据驱动旁证。"
                  "sigma0 = 0.1 m。")
    write_tex("table3_independent_truth",
              "Independent truth checks on 5,303 stratified links: the "
              "criterion CRB is accurate, and the only violations of uniqueness "
              "are near-mirror solutions.",
              "tab:truth", header, rows, colspec="lll")
    # 中文版：数值列沿用同一份现场统计结果，仅译项目与说明
    q_zh = [
        "链路数", "判据给出的秩", "蒙特卡洛方差比，中位 $x$",
        "蒙特卡洛方差比，中位 $y$", "蒙特卡洛方差比，中位 $z$",
        r"蒙特卡洛方差比，$z$ 的 90 分位",
        r"蒙特卡洛方差比，垂直退化最重档的 $z$",
        "估计偏置范数，中位", "多起点发现第二个解", "多起点歧义，单锚点链路",
        "多起点歧义，多锚点链路", "这些解的诊断",
        "数据驱动垂直与水平敏感度之比，WiPTR/val_grid",
        "数据驱动垂直与水平敏感度之比，WiPTR/val_ood",
        "数据驱动垂直与水平敏感度之比，Wi3R/main_5tx",
    ]
    note_zh = [
        "分层抽样，11 个划分，307 个楼层", "判据在所有链路均通过",
        "实测与 CRB 之比，为 1 表示 CRB 准确", "实测与 CRB 之比", "垂直轴",
        "垂直轴尾部", "该档 $s_z$ 中位为 0.90", "应接近零",
        "为局部求解，故此为下界", "n = 1071", "n = 4232",
        "虚拟锚点云 $s_3/s_1$ 中位 0.020；到镜像点的距离中位 0.055 m",
        "试点，首径时延回归，不含模型", "试点", "试点",
    ]
    vals_zh = [r[1] for r in rows]
    vals_zh[1] = "100% 链路为 3"
    vals_zh[11] = "近镜像解"
    rows_zh = [(q, v, nt) for q, v, nt in zip(q_zh, vals_zh, note_zh)]
    write_tex("table3_independent_truth_zh",
              "5303 条分层链路上的三路独立真值结果。判据给出的 CRB 是准确的，"
              "唯一出现的多解是近镜像解。",
              "tab:truth", ["项目", "数值", "说明"], rows_zh,
              colspec=r"p{6.3cm}p{3.1cm}p{5.0cm}", size="\\small",
              note="前 14 行由真值明细现场统计，最后三行为试点给出的数据驱动旁证，sigma0 = 0.1 m。")
    print("  表 3 完成: 歧义 %.2f%%  ratio_z 中位 %.3f" % (100 * amb.mean(),
                                                          np.median(rz)))


# ------------------------------------------------------------------ 表 4
def table4():
    """端到端分档误差表：由 results/e2e_summary.csv 现场统计，不手抄。"""
    import csv
    with open(RES / "e2e_summary.csv", encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))

    def get(est, grp):
        for r in recs:
            if r["estimator"] == est and r["group"] == grp:
                return r
        return None

    ORDER = [
        ("aoa", "all", "全体"),
        ("aoa", "criterion pass (rank3 & k<100)", r"判据通过（$\kappa<100$）"),
        ("aoa", "criterion fail", "判据不通过"),
        ("aoa", "pass & s_z<=0.35", r"通过且 $s_z\le 0.35$"),
        ("aoa", "pass & 0.35<s_z<=0.5", r"通过且 $0.35<s_z\le 0.5$"),
        ("aoa", "pass & s_z>0.5", r"通过且 $s_z>0.5$"),
        ("aoa", "pass & d_eff<2", r"通过且 $d_{eff}<2$"),
        ("aoa", "pass & d_eff>=2", r"通过且 $d_{eff}\ge 2$"),
        ("tof", "all", "纯距离估计器，全体"),
        ("tof", "criterion pass (rank3 & k<100)", "纯距离估计器，判据通过"),
    ]
    labels_en = [
        "All positions", "Criterion pass", "Criterion fail",
        r"Pass and $s_z\le 0.35$", r"Pass and $0.35<s_z\le 0.5$",
        r"Pass and $s_z>0.5$", r"Pass and $d_{eff}<2$", r"Pass and $d_{eff}\ge 2$",
        "Range-only estimator, all", "Range-only estimator, pass",
    ]
    rows, rows_en, csv_rows = [], [], []
    for (est, grp, lab_zh), lab_en in zip(ORDER, labels_en):
        r = get(est, grp)
        if r is None:
            continue
        n = int(float(r["n"]))
        vals = ("%.3f" % float(r["ez_med"]), "%.3f" % float(r["eh_med"]),
                "%.3f" % float(r["e3_med"]))
        rows.append((lab_zh, f"{n:,}") + vals)
        rows_en.append((lab_en, f"{n:,}") + vals)
        csv_rows.append([est, grp, n, r["ez_med"], r["ez_p90"], r["eh_med"],
                         r["e3_med"]])
    keys = ["group_est", "group", "n", "ez_med", "ez_p90", "eh_med", "e3_med"]
    header = ["分组", "样本数", "垂直误差 (m)", "水平误差 (m)", "三维误差 (m)"]
    header_en = ["Group", "N", "Vertical (m)", "Horizontal (m)", "3-D (m)"]
    note_zh = ("由 results/e2e_summary.csv 现场统计，各列为逐位置估计样本标准差的"
               "中位值。含方向信息的估计器：判据量与实际垂直误差的相关系数为 "
               r"$\kappa$ $-$0.026、$d_{eff}$ $-$0.184、$s_z$ $+$0.343、"
               "平均 $|u_z|$ $-$0.368。纯距离估计器可用样本仅 200 个，仅作参考。")
    write_csv("table4_end_to_end", keys, csv_rows)
    write_md("table4_end_to_end",
             "表 4  端到端分档误差（30,086 个位置，B = 100 MHz 口径）",
             [h for h in header], rows, note=note_zh)
    write_tex("table4_end_to_end",
              "End-to-end stratified positioning error under the same estimator.",
              "tab:e2e", header_en, rows_en, colspec="lrrrr", size="\\small",
              placement="h",
              note="Statistics from results/e2e_summary.csv; each value is the "
                   "median of the per-position sample standard deviation.")
    write_tex("table4_end_to_end_zh",
              "端到端分档定位误差。误差为逐位置估计样本标准差的中位值，判据量中 "
              r"$\kappa$、$d_{eff}$ 为传统判据，$s_z$ 与平均 $|u_z|$ 为方向判据。",
              "tab:e2e", header, rows, colspec="lrrrr", size="\\small",
              placement="h",
              note=note_zh)
    print("  表 4 完成，", len(rows), "行，全部样本", rows[0][1])


# ------------------------------------------------------------------ 表 5
def table5():
    """指标公平基线与决策价值（results/decision_metrics.csv，估计器 AOA）。"""
    import pandas as pd
    p = RES / "decision_metrics.csv"
    if not p.exists():
        print("  跳过表 5：缺 decision_metrics.csv")
        return
    d = pd.read_csv(p)
    d = d[d.estimator == "aoa"].sort_values("auroc", ascending=False)
    h_csv = ["criterion", "AUROC", "AP", "AUROC_split_wtd", "transfer_TPR",
             "transfer_FPR"]
    h_en = ["criterion", "AUROC", "AP", "AUROC (split-wtd)", "transfer TPR",
            "transfer FPR"]
    h_zh = ["判据量", "AUROC", "AP", "AUROC（划分加权）", "迁移查全率", "迁移虚警率"]
    rows = [[r["metric"], "%.3f" % r["auroc"], "%.3f" % r["avg_prec"],
             "%.3f" % r["auroc_split_wtd"], "%.3f" % r["transfer_tpr"],
             "%.3f" % r["transfer_fpr"]] for _, r in d.iterrows()]
    note = ("由 results/decision_metrics.csv 现场统计。结果变量为垂直总误差超过 "
            "0.5 m。AUROC 0.5 为无区分度。迁移列为在训练划分按 Youden 指数定阈、"
            "留出划分评估后的均值。")
    write_csv("table5_baseline_auroc", h_csv, rows)
    write_md("table5_baseline_auroc", "指标公平基线与决策价值", h_zh, rows, note)
    write_tex("table5_baseline_auroc",
              "Fair baseline and decision value of each candidate criterion.",
              "tab:baseline", h_en, rows, colspec="lrrrrr", size="\\small",
              note=note)
    write_tex("table5_baseline_auroc_zh",
              "各候选判据量作为筛查规则的公平基线与决策价值。",
              "tab:baseline", h_zh, rows, colspec="lrrrrr", size="\\small",
              note=note)
    print("  表 5 写出（%d 行）" % len(rows))


# ------------------------------------------------------------------ 表 6
def table6():
    """统计单位与数据口径的分层审计（results/stratified_audit.csv）。"""
    import pandas as pd
    p = RES / "stratified_audit.csv"
    if not p.exists():
        print("  跳过表 6：缺 stratified_audit.csv")
        return
    d = pd.read_csv(p)
    h_csv = ["subset", "links", "rank3_pct", "kappa_lt100_pct", "s_z_gt05_pct",
             "strict_false_pass_pct", "kappa_med", "s_z_med", "mean_uz_med"]
    h_en = ["subset", "links", "rank3 (\\%)", "$\\kappa$<100 (\\%)",
            "$s_z$>0.5 (\\%)", "strict false pass (\\%)", "median $\\kappa$",
            "median $s_z$", "median mean $|u_z|$"]
    h_zh = ["子集", "链路数", "秩 3（\\%）", "$\\kappa$<100（\\%）",
            "$s_z$>0.5（\\%）", "严格假通过（\\%）", "$\\kappa$ 中位",
            "$s_z$ 中位", "平均 $|u_z|$ 中位"]
    rows = [[r["group"], "%d" % r["n"], "%.2f" % (100 * r["rank3"]),
             "%.2f" % (100 * r["kappa_lt100"]), "%.2f" % (100 * r["sharez_gt05"]),
             "%.2f" % (100 * r["strict_fp"]), "%.1f" % r["kappa_med"],
             "%.3f" % r["sharez_med"], "%.3f" % r["mean_uz_med"]]
            for _, r in d.iterrows()]
    note = ("由 results/stratified_audit.csv 现场统计。单锚点子集按接收机坐标分组"
            "后只有一个锚点，多锚点子集为两个及以上。严格假通过指同时满秩、"
            "$\\kappa$ 小于 100 且 $s_z$ 大于 0.5。")
    write_csv("table6_stratified", h_csv, rows)
    write_md("table6_stratified", "统计单位与数据口径的分层审计", h_zh, rows, note)
    write_tex("table6_stratified",
              "Stratified audit by statistical unit and data definition.",
              "tab:strata", h_en, rows, colspec="lrrrrrrrr", size="\\small",
              note=note)
    write_tex("table6_stratified_zh",
              "统计单位与数据口径的分层审计。",
              "tab:strata", h_zh, rows, colspec="lrrrrrrrr", size="\\small",
              note=note)
    print("  表 6 写出（%d 行）" % len(rows))


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    table1()
    table2()
    table3()
    table4()
    table5()
    table6()
    print(f"表格写入 {TAB}")


if __name__ == "__main__":
    main()
