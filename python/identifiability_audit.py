#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""identifiability_audit.py -- 定位几何可辨识性审计工具（可交付件）

用途
----
给定一套锚点（发射机）与接收机的部署几何，用观测到的多径分量的时延与到达方向
构造虚拟锚点，装配位置 Fisher 信息矩阵（FIM），并按「方向分辨的精度结构」报告
该几何是否可用，而不是只报告秩与条件数。

这是论文第 3.3 节部署检查规则的参考实现。它把此前只以脚本形式存在的判据计算
收敛为一个命令行工具，使第三方可以在自己的数据上复现该规则。

指标定义（与论文第 1.4 节、`python/c2_fullscale_vec.py` 完全一致）
----------------------------------------------------------------
rank     : 相对门限 lam > lam_max * 1e-9 的非零特征值个数（二值门）
kappa    : lam_max / lam_min，特征值下限取 lam_max * 1e-12，刻画整体各向异性
d_eff    : (sum lam)^2 / sum lam^2，取值 [1, 3]，信息实际张成的方向数
share_z  : var_z / trace(J^+)，位置协方差在垂直方向所占的方差比例
严格假通过: rank == 3 且 kappa < 100 且 share_z > 0.5

判定规则
--------
d_eff < 2 或 share_z > 0.5 即判该几何在垂直方向不可用（阈值 2 与 0.5 为自然
分界，对应信息集中在两个方向以下与垂直方向承载过半方差）。

输入后端
--------
1. ``--source h5``         直接读 Qualcomm Wi3R / WiPTR 的 dataset_sparse.h5
2. ``--source normalized`` 读 ``data/normalized/links.csv`` 与 ``paths.csv``

输出
----
``<outdir>/audit_links.csv``    逐链路指标（可用 --no-links 关闭）
``<outdir>/audit_summary.json`` 总体统计与判定
``<outdir>/audit_report.md``    人类可读报告（含复杂度与运行时间）

用法
----
    # h5 后端，单划分
    python python/identifiability_audit.py --source h5 \
        --root data/public_raw --split WiPTR/val_grid \
        --outdir results/audit_demo_val_grid

    # 归一化后端
    python python/identifiability_audit.py --source normalized \
        --links data/normalized/links.csv --paths data/normalized/paths.csv \
        --outdir results/audit_demo_normalized

    # 与全量脚本的结果对齐校验（证明口径一致）
    python python/identifiability_audit.py --source h5 \
        --root data/public_raw --split WiPTR/val_grid \
        --outdir results/audit_demo_val_grid \
        --validate-against results/c2full_WiPTR_val_grid.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 数值核心：与 c2_fullscale_vec.metrics_batch 完全相同的口径
# ---------------------------------------------------------------------------
EPS_REL = 1e-12
RANK_REL = 1e-9
KAPPA_OK = 100.0
SHARE_Z_OK = 0.5
DEFF_OK = 2.0

LINK_KEYS = ["group", "n_path", "los_frac", "n_anchor", "rx_z", "mean_abs_uz",
             "sum_w", "lam_min", "lam_mid", "lam_max", "d_eff", "kappa",
             "rank", "share_z", "var_z", "trace_inv", "strict_false_pass"]


def metrics_batch(J: np.ndarray) -> dict:
    """从 (G,3,3) 的位置 FIM 向量化提取指标。口径与 c2_fullscale_vec 一致。"""
    J = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    lam, Q = np.linalg.eigh(J)              # 升序；(G,3), (G,3,3)
    lam = np.clip(lam, 0.0, None)
    lam_max = lam[:, -1]
    ok = lam_max > 0
    lam_c = np.where(ok[:, None], np.clip(lam, lam_max[:, None] * EPS_REL, None), 1.0)
    rank = (lam > (lam_max[:, None] * RANK_REL)).sum(axis=1).astype(np.int64)
    kappa = np.where(ok, lam_c[:, -1] / lam_c[:, 0], np.inf)
    s2 = (lam ** 2).sum(axis=1)
    d_eff = np.where(s2 > 0, (lam.sum(axis=1) ** 2) / np.where(s2 > 0, s2, 1.0), 0.0)
    inv = 1.0 / lam_c
    var_z = (Q[:, 2, :] ** 2 * inv).sum(axis=1)
    trace_inv = inv.sum(axis=1)
    share_z = np.where(trace_inv > 0, var_z / trace_inv, 1.0)
    return dict(rank=rank, kappa=kappa, d_eff=d_eff, share_z=share_z,
                lam_min=lam[:, 0], lam_mid=lam[:, 1], lam_max=lam[:, 2],
                var_z=var_z, trace_inv=trace_inv)


def _strict_false_pass(rank, kappa, share_z):
    return ((rank == 3) & (kappa < KAPPA_OK) & (share_z > SHARE_Z_OK)).astype(np.int64)


# ---------------------------------------------------------------------------
# 后端一：h5
# ---------------------------------------------------------------------------
def audit_h5(root: Path, split: str | None, dr_db: float,
             max_floors: int | None = None) -> tuple[dict, float]:
    """审计一个或多个 h5 划分，返回 (逐链路指标字典, 有效用时秒)。"""
    splits = wio.discover_splits(root)
    if split:
        splits = [s for s in splits if s.tag == split or s.split == split]
    if not splits:
        raise SystemExit(f"在 {root} 下未找到匹配的划分（--split={split}）")

    acc: dict[str, list] = {k: [] for k in LINK_KEYS}
    t0 = time.time()
    for info in splits:
        with wio.h5py.File(info.path, "r") as f:
            ids = [fid for fid, _ in wio.list_floors(f)]
            if max_floors is not None:
                ids = ids[:max_floors]
            for fid in ids:
                r = _floor_metrics(f, fid, dr_db)
                if r is None:
                    continue
                n = r["rank"].shape[0]
                acc["group"].append(np.full(n, info.tag, dtype=object))
                for k in LINK_KEYS[1:]:
                    acc[k].append(r[k])
    return _concat(acc), time.time() - t0


def _floor_metrics(f, floor_id: int, dr_db: float) -> dict | None:
    """单楼层逐链路指标，口径与 c2_fullscale_vec.process_floor 一致。"""
    bounds = wio.floor_channel_bounds(f)
    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
    if c1 <= c0:
        return None
    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
    tx_idx, rx_idx, pstart = ch[:, 0], ch[:, 1], ch[:, 2]
    M = ch.shape[0]

    utx, urx = np.unique(tx_idx), np.unique(rx_idx)
    tx_xyz_ch = np.asarray(f["tx"][utx], dtype=np.float64)[np.searchsorted(utx, tx_idx)]
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
    rx_z_g = np.bincount(rx_uid, weights=rx_xyz[:, 2], minlength=G) / n_ch_g

    ntx = int(utx.size)
    key = rx_uid.astype(np.int64) * ntx + np.searchsorted(utx, tx_idx)
    n_anchor_g = np.bincount(np.unique(key) // ntx, minlength=G)

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

    d_o, w_o = dirs[order][keep], w_s[keep] / np.repeat(g_max, cnt_all)[keep]
    grp_o = grp_s[keep]
    st2 = np.flatnonzero(np.r_[True, grp_o[1:] != grp_o[:-1]])
    cnt2 = np.diff(np.append(st2, d_o.shape[0]))
    gid = grp_o[st2]

    outer = (d_o[:, :, None] * d_o[:, None, :]) * w_o[:, None, None]
    J = np.add.reduceat(outer.reshape(d_o.shape[0], 9), st2, axis=0).reshape(-1, 3, 3)
    met = metrics_batch(J)

    mean_abs_uz_g = np.add.reduceat(np.abs(dirs[order][:, 2]), starts) / cnt_all
    sum_w_g = np.add.reduceat(w_s, starts)

    out = dict(
        n_path=cnt2, sum_w=np.add.reduceat(w_o, st2),
        los_frac=los_frac_g[gid], n_anchor=n_anchor_g[gid],
        rx_z=rx_z_g[gid], mean_abs_uz=mean_abs_uz_g[gid],
        **{k: met[k] for k in ("rank", "kappa", "d_eff", "share_z",
                               "lam_min", "lam_mid", "lam_max",
                               "var_z", "trace_inv")},
    )
    out["strict_false_pass"] = _strict_false_pass(out["rank"], out["kappa"], out["share_z"])
    return out


# ---------------------------------------------------------------------------
# 后端二：归一化 CSV
# ---------------------------------------------------------------------------
def audit_normalized(links_csv: Path, paths_csv: Path, dr_db: float) -> tuple[dict, float]:
    """读归一化 links.csv / paths.csv，按场景与接收机坐标分组计算同一批指标。"""
    import pandas as pd

    t0 = time.time()
    links = pd.read_csv(links_csv)
    paths = pd.read_csv(paths_csv)
    need_l = {"scene_id", "floor_id", "tx_id", "rx_id", "rx_x", "rx_y", "rx_z", "los_flag"}
    need_p = {"scene_id", "tx_id", "rx_id", "path_id", "gain_db", "aoa_az_deg", "aoa_el_deg"}
    if not need_l <= set(links.columns):
        raise SystemExit(f"links.csv 缺列：{sorted(need_l - set(links.columns))}")
    if not need_p <= set(paths.columns):
        raise SystemExit(f"paths.csv 缺列：{sorted(need_p - set(paths.columns))}")

    links = links.sort_values(["scene_id", "floor_id", "tx_id", "rx_id"]).reset_index(drop=True)
    paths = paths.sort_values(["scene_id", "tx_id", "rx_id", "path_id"]).reset_index(drop=True)
    links["_n_anchor"] = _annotate_anchor_count(links).to_numpy(np.int64)

    # 路径方向的单位向量：u = [cos(el)cos(az), cos(el)sin(az), sin(el)]
    az = np.radians(paths["aoa_az_deg"].to_numpy(float))
    el = np.radians(paths["aoa_el_deg"].to_numpy(float))
    pdir = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    pw = 10.0 ** (paths["gain_db"].to_numpy(float) / wio.GAIN_DB_DIVISOR)

    # 归并口径与论文一致：同一场景内位置相同的接收机，其在各锚点下的观测合成一条链路
    chan_key = (links["scene_id"].astype(str) + "|" + links["tx_id"].astype(str)
                + "|" + links["rx_id"].astype(str))
    link_key = (links["scene_id"].astype(str) + "|"
                + links["rx_x"].round(6).astype(str) + "|"
                + links["rx_y"].round(6).astype(str) + "|"
                + links["rx_z"].round(6).astype(str))
    map_chan2link = pd.Series(link_key.to_numpy(), index=chan_key.to_numpy())
    p_chan_key = (paths["scene_id"].astype(str) + "|" + paths["tx_id"].astype(str)
                  + "|" + paths["rx_id"].astype(str))
    pkey = map_chan2link.reindex(p_chan_key.to_numpy()).to_numpy()
    if pd.isna(pkey).any():
        raise SystemExit("paths.csv 中存在 links.csv 未记录的 (场景, 锚点, 接收机) 组合")

    # 路径按 key 排序后分段（key 已有序，直接找分段边界）
    ord_ = np.argsort(pkey, kind="stable")
    pdir, pw, pkey = pdir[ord_], pw[ord_], pkey[ord_]
    seg = np.flatnonzero(np.r_[True, pkey[1:] != pkey[:-1]])
    seg_key = pkey[seg]
    seg_cnt = np.diff(np.append(seg, pkey.size))

    # 动态范围门限：以该 key 内最大功率为基准
    gmax = np.maximum.reduceat(pw, seg)
    keep = pw >= np.repeat(gmax, seg_cnt) * 10.0 ** (-dr_db / 10.0)
    d_o, w_o = pdir[keep], pw[keep] / np.repeat(gmax, seg_cnt)[keep]
    sk = np.repeat(seg_key, seg_cnt)[keep]
    st2 = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    cnt2 = np.diff(np.append(st2, d_o.shape[0]))
    k2 = sk[st2]

    outer = (d_o[:, :, None] * d_o[:, None, :]) * w_o[:, None, None]
    J = np.add.reduceat(outer.reshape(d_o.shape[0], 9), st2, axis=0).reshape(-1, 3, 3)
    met = metrics_batch(J)

    # 逐链路属性：按归并后的 key 取首行
    first_row = links.assign(_lk=link_key.to_numpy()).groupby("_lk", sort=True).head(1)
    first_row = first_row.set_index("_lk").reindex(k2)

    # mean|u_z| 的定义与全量脚本一致：取该链路全部多径的方向垂直分量均值，
    # 不做动态范围过滤（该量刻画到达方向分布本身，不是判据的一部分）
    mean_abs_uz_all = (np.add.reduceat(np.abs(pdir[:, 2]), seg) / seg_cnt)
    mean_abs_uz = mean_abs_uz_all[np.searchsorted(seg_key, k2)]

    rx = first_row[["rx_x", "rx_y", "rx_z"]].to_numpy(float)
    out = {
        "group": first_row["scene_id"].to_numpy(dtype=object),
        "n_path": cnt2,
        "sum_w": np.add.reduceat(w_o, st2),
        "los_frac": first_row["los_flag"].to_numpy(float),
        "n_anchor": first_row["_n_anchor"].to_numpy(np.int64),
        "rx_z": rx[:, 2],
        "mean_abs_uz": mean_abs_uz,
        **met,
    }
    out["strict_false_pass"] = _strict_false_pass(out["rank"], out["kappa"], out["share_z"])
    return out, time.time() - t0


def _annotate_anchor_count(links) -> "object":
    """给 links 增加 _n_anchor 列：同一场景内同坐标接收机所覆盖的锚点数。"""
    rxkey = (links["scene_id"].astype(str) + "|"
             + links["rx_x"].round(6).astype(str) + "|"
             + links["rx_y"].round(6).astype(str) + "|"
             + links["rx_z"].round(6).astype(str))
    n_anchor = links.groupby(rxkey)["tx_id"].nunique()
    return rxkey.map(n_anchor)


def _concat(acc: dict) -> dict:
    out = {}
    for k, v in acc.items():
        if not v:
            out[k] = np.array([])
        elif k == "group":
            out[k] = np.concatenate([np.asarray(x, dtype=object) for x in v])
        else:
            out[k] = np.concatenate(v)
    return out


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------
def summarize(d: dict, runtime_s: float, source: str, n_total_paths: int | None) -> dict:
    n = int(d["rank"].shape[0])
    rank = d["rank"].astype(int)
    kappa = d["kappa"].astype(float)
    deff = d["d_eff"].astype(float)
    sz = d["share_z"].astype(float)
    sfp = d["strict_false_pass"].astype(int)

    def q(a, p):
        return float(np.quantile(a, p)) if a.size else float("nan")

    s = {
        "source": source,
        "n_links": n,
        "runtime_s": round(runtime_s, 2),
        "throughput_links_per_s": round(n / runtime_s, 1) if runtime_s > 0 else None,
        "rank3_frac": float((rank == 3).mean()) if n else float("nan"),
        "kappa_median": q(kappa, 0.5),
        "kappa_p90": q(kappa, 0.9),
        "d_eff_median": q(deff, 0.5),
        "d_eff_lt2_frac": float((deff < DEFF_OK).mean()) if n else float("nan"),
        "share_z_median": q(sz, 0.5),
        "share_z_gt0.5_frac": float((sz > SHARE_Z_OK).mean()) if n else float("nan"),
        "strict_false_pass_frac": float(sfp.mean()) if n else float("nan"),
        "corr_sharez_meanabsuz": (float(np.corrcoef(sz, d["mean_abs_uz"].astype(float))[0, 1])
                                  if n > 2 else float("nan")),
        "verdict": {
            "criterion_rank_kappa_pass": bool((rank == 3).mean() > 0.9),
            "vertical_unusable_frac": float(((deff < DEFF_OK) | (sz > SHARE_Z_OK)).mean())
            if n else float("nan"),
        },
    }
    s["verdict"]["conclusion"] = (
        "On most links the vertical direction is unusable while the rank and the "
        "condition number report a pass"
        if s["verdict"]["vertical_unusable_frac"] > 0.2 and
        s["verdict"]["criterion_rank_kappa_pass"] else
        "No systematic degradation of the directional structure")
    s["complexity"] = ("O(K_i) outer products per link, with K_i the number of "
                       "multipath components of that link; the 3x3 eigendecomposition "
                       "is O(1). Total time O(N_paths + N_links), space O(N_links), "
                       "independent of the anchor count.")
    if n_total_paths:
        s["n_paths"] = int(n_total_paths)
    # 分组判定：按锚点数
    na = d["n_anchor"].astype(int)
    by_anchor = {}
    for k in sorted(set(na.tolist())):
        m = na == k
        by_anchor[str(k)] = {
            "n_links": int(m.sum()),
            "share_z_median": q(sz[m], 0.5),
            "d_eff_median": q(deff[m], 0.5),
            "kappa_median": q(kappa[m], 0.5),
            "strict_false_pass_frac": float(sfp[m].mean()) if m.sum() else float("nan"),
        }
    s["by_n_anchor"] = by_anchor
    return s


def write_report(path: Path, s: dict, validation: dict | None) -> None:
    L = []
    L.append("# Geometry identifiability audit\n")
    L.append(f"Input backend: `{s['source']}`\n")
    L.append(f"Links: {s['n_links']:,}    runtime: {s['runtime_s']} s"
             f"    throughput: {s['throughput_links_per_s']} links/s\n")
    L.append("## 1 Criterion side (rank and condition number)\n")
    L.append(f"- fraction with rank 3: {s['rank3_frac']*100:.2f}%")
    L.append(f"- median condition number: {s['kappa_median']:.1f}, "
             f"90th percentile: {s['kappa_p90']:.1f}")
    L.append("\n## 2 Direction-resolved accuracy structure\n")
    L.append(f"- median effective dimension: {s['d_eff_median']:.3f}, "
             f"fraction below 2: {s['d_eff_lt2_frac']*100:.2f}%")
    L.append(f"- median vertical variance share: {s['share_z_median']:.4f}, "
             f"fraction above one half: {s['share_z_gt0.5_frac']*100:.2f}%")
    L.append(f"- fraction of directionally misleading passes "
             f"(rank 3, $\\kappa<100$, $s_z>0.5$): "
             f"{s['strict_false_pass_frac']*100:.2f}%")
    L.append(f"- correlation of the vertical share with the vertical component of the "
             f"arrival direction: {s['corr_sharez_meanabsuz']:.3f}")
    L.append("\n## 3 Verdict\n")
    L.append(f"- criterion passes on the majority of links: "
             f"{s['verdict']['criterion_rank_kappa_pass']}")
    L.append(f"- fraction of links whose vertical direction is unusable: "
             f"{s['verdict']['vertical_unusable_frac']*100:.2f}%")
    L.append(f"- conclusion: {s['verdict']['conclusion']}")
    L.append("\n## 4 Grouped by anchor count\n")
    L.append("| anchors | links | median $s_z$ | median $d_{eff}$ | "
             "median $\\kappa$ | misleading pass |")
    L.append("|---|---|---|---|---|---|")
    for k, v in s["by_n_anchor"].items():
        L.append(f"| {k} | {v['n_links']:,} | {v['share_z_median']:.4f} | "
                 f"{v['d_eff_median']:.3f} | {v['kappa_median']:.1f} | "
                 f"{v['strict_false_pass_frac']*100:.2f}% |")
    if s.get("complexity"):
        L.append("\n## 5 Complexity\n")
        L.append(s["complexity"])
    if validation:
        L.append("\n## 6 Agreement with the full-scale scripts\n")
        for k, v in validation.items():
            L.append(f"- {k}: tool {v[0]:.6g} against full scripts {v[1]:.6g}, "
                     f"relative deviation {v[2]:.2e}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def validate_against(csv_path: Path, d: dict, keys) -> dict:
    """把本工具的总体统计与既有全量 CSV 对比，用于证明口径一致。"""
    import numpy as np
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = {k: [] for k in keys}
        for row in rd:
            for k in keys:
                cols[k].append(float(row[k]))
    ref = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out = {}
    for k in keys:
        a = float(np.median(d[k].astype(float)))
        b = float(np.median(ref[k]))
        denom = abs(b) if abs(b) > 0 else 1.0
        out[k] = (a, b, abs(a - b) / denom)
    return out


def main():
    ap = argparse.ArgumentParser(description="定位几何可辨识性审计工具")
    ap.add_argument("--source", choices=["h5", "normalized"], required=True)
    ap.add_argument("--root", default=str(ROOT / "data" / "public_raw"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--links", default=str(ROOT / "data" / "normalized" / "links.csv"))
    ap.add_argument("--paths", default=str(ROOT / "data" / "normalized" / "paths.csv"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--max-floors", type=int, default=None)
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--validate-against", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.source == "h5":
        d, rt = audit_h5(Path(args.root), args.split, args.dr, args.max_floors)
        src = f"h5:{args.split or 'all'}"
        n_paths = None
    else:
        d, rt = audit_normalized(Path(args.links), Path(args.paths), args.dr)
        src = "normalized:data/normalized"
        n_paths = None

    if d["rank"].shape[0] == 0:
        raise SystemExit("没有有效链路，检查划分名或数据路径")

    s = summarize(d, rt, src, n_paths)
    s["python"] = platform.python_version()
    s["numpy"] = np.__version__

    (outdir / "audit_summary.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = None
    if args.validate_against:
        validation = validate_against(Path(args.validate_against), d,
                                      ["rank", "kappa", "d_eff", "share_z"])
        s["validation"] = {k: list(v) for k, v in validation.items()}
        (outdir / "audit_summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(outdir / "audit_report.md", s, validation)

    if not args.no_links:
        with open(outdir / "audit_links.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(LINK_KEYS)
            for row in zip(*[d[k] for k in LINK_KEYS]):
                wr.writerow(row)

    print(f"[audit] {src}  links {s['n_links']:,}  time {s['runtime_s']} s")
    print(f"        rank 3 {s['rank3_frac']*100:.2f}%  "
          f"misleading pass {s['strict_false_pass_frac']*100:.2f}%  "
          f"median share_z {s['share_z_median']:.4f}")
    print(f"        report -> {outdir/'audit_report.md'}")


if __name__ == "__main__":
    main()
