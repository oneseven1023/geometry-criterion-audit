#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fingerprint_baseline.py -- 传统定位基线：指纹 KNN 与 ToF 几何定位

读取 ``normalize_rt.py`` 产出的 ``links.csv``，在统一接口下运行多条基线，
输出误差统计与逐位置预测，用于论文中的泛化对比。

基线
----
``knn_rss``       仅用 RSS 向量的 KNN 指纹
``knn_rss_tof``   RSS + 首径时延 的 KNN 指纹
``knn_rss_mp``    RSS + 多径统计（时延扩展、路径数、到达角）的 KNN 指纹
``tof_trilat``    仅用首径时延的几何多边定位（非线性最小二乘）

评估模式
--------
``within``  同场景内按位置划分训练/测试，考察空间外推（对应实验 E1）
``cross``   留一场景交叉验证，用其余场景训练、当前场景测试（对应实验 E2）
``pooled``  同场景内随机划分，作为上界参照

用法
----
    python python/fingerprint_baseline.py --input data/normalized/links.csv \
        --outdir results --k 5 --mode within

    # 指定测试场景
    python python/fingerprint_baseline.py --input data/normalized/links.csv \
        --outdir results --test-scenes WiPTR-val_ood-00000,WiPTR-val_ood-00001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

C_M_PER_NS = 0.299792458

FEATURE_SETS = {
    "rss": ["rss_db"],
    "rss_tof": ["rss_db", "tof_ns"],
    "rss_mp": ["rss_db", "delay_spread_ns", "path_count", "aoa_az_deg", "aoa_el_deg"],
    "tof": ["tof_ns"],
}


# --------------------------------------------------------------------------- #
# 特征构造
# --------------------------------------------------------------------------- #
def build_fingerprints(df: pd.DataFrame, feature_cols: list[str],
                       n_anchors: int | None = None,
                       anchor_list: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """把长表 ``links.csv`` 透视成「一行一个接收机位置」的指纹矩阵。

    Returns
    -------
    X : (N, n_anchor * n_feature) 指纹矩阵
    y : (N, 3) 真实坐标
    index : 每行的 (scene_id, rx_id, tx 列表) 元信息
    """
    rows = []
    Xs = []
    ys = []
    for scene_id, g in df.groupby("scene_id", sort=True):
        piv = {}
        tx_ids = sorted(g["tx_id"].unique())
        if anchor_list is not None:
            tx_ids = [t for t in tx_ids if t in set(anchor_list)]
        if n_anchors is not None and len(tx_ids) != n_anchors:
            continue
        ok = True
        for t in tx_ids:
            gt = g[g["tx_id"] == t].sort_values("rx_id")
            for c in feature_cols:
                piv[(t, c)] = gt.set_index("rx_id")[c]
        base = g[g["tx_id"] == tx_ids[0]].sort_values("rx_id")
        rx_ids = base["rx_id"].to_numpy()
        for t in tx_ids[1:]:
            gt = g[g["tx_id"] == t].sort_values("rx_id")
            if not np.array_equal(gt["rx_id"].to_numpy(), rx_ids):
                ok = False
                break
        if not ok:
            continue
        mat = np.column_stack([piv[(t, c)].to_numpy(dtype=np.float64)
                               for t in tx_ids for c in feature_cols])
        finite = np.isfinite(mat).all(axis=1)
        mat = mat[finite]
        rid = rx_ids[finite]
        pos = base.set_index("rx_id").loc[rid, ["rx_x", "rx_y", "rx_z"]].to_numpy(dtype=np.float64)
        Xs.append(mat)
        ys.append(pos)
        rows.append(pd.DataFrame({
            "scene_id": scene_id, "rx_id": rid,
            "n_anchor": len(tx_ids),
            "tx_ids": ",".join(str(t) for t in tx_ids),
        }))
    if not Xs:
        return np.empty((0, 0)), np.empty((0, 3)), pd.DataFrame()
    return np.vstack(Xs), np.vstack(ys), pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 基线实现
# --------------------------------------------------------------------------- #
class KnnFingerprint:
    """特征空间欧氏距离 KNN 指纹定位。"""

    def __init__(self, k: int = 5, weighted: bool = True,
                 standardize: str = "none") -> None:
        self.k = k
        self.weighted = weighted
        self.standardize = standardize

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KnnFingerprint":
        self.mu_ = X.mean(axis=0) if self.standardize != "none" else np.zeros(X.shape[1])
        sd = X.std(axis=0) if self.standardize != "none" else np.ones(X.shape[1])
        sd[sd < 1e-9] = 1.0
        self.sd_ = sd
        self.Xt_ = (X - self.mu_) / self.sd_
        self.y_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        out = np.empty((Z.shape[0], 3))
        for i in range(Z.shape[0]):
            d = np.linalg.norm(self.Xt_ - Z[i], axis=1)
            k = min(self.k, d.size)
            idx = np.argpartition(d, k - 1)[:k]
            if self.weighted:
                w = 1.0 / np.maximum(d[idx], 1e-6)
            else:
                w = np.ones(k)
            out[i] = (self.y_[idx] * w[:, None]).sum(axis=0) / w.sum()
        return out


def tof_trilaterate(test_df: pd.DataFrame, anchors: pd.DataFrame,
                    dims: int | None = None) -> np.ndarray:
    """仅用首径时延做几何多边定位。

    对每个接收机位置，在锚点集上最小化
        sum_i ( ||p - a_i|| - c * tof_i )^2
    以该位置在各锚点上的加权质心作为初值。

    维数自动选择：若该场景所有接收机 z 坐标一致（高差 < 0.05 m），按二维求解，
    至少需要 3 个锚点；否则按三维求解，至少需要 4 个锚点。锚点数不足时返回 NaN，
    这是几何可观测性的硬约束，不是实现缺陷。
    """
    a_xyz = anchors[["tx_x", "tx_y", "tx_z"]].to_numpy(dtype=np.float64)
    a_id = anchors["tx_id"].to_numpy()
    if dims is None:
        dims = 2 if float(test_df["rx_z"].ptp()) < 0.05 else 3

    preds = []
    for _, g in test_df.groupby("rx_id", sort=True):
        g = g.set_index("tx_id")
        common = [t for t in a_id if t in g.index]
        if len(common) < dims + 1:
            preds.append([np.nan, np.nan, np.nan])
            continue
        idx = [list(a_id).index(t) for t in common]
        ai = a_xyz[idx][:, :dims]
        di = g.loc[common, "tof_ns"].to_numpy(dtype=np.float64) * C_M_PER_NS
        if not np.isfinite(di).all() or (di <= 0).any():
            preds.append([np.nan, np.nan, np.nan])
            continue

        def resid(p):
            return np.linalg.norm(p[None, :] - ai, axis=1) - di

        x0 = np.average(ai, axis=0, weights=1.0 / np.maximum(di, 1e-3))
        try:
            r = least_squares(resid, x0, method="lm", max_nfev=400)
            out = np.full(3, np.nan)
            out[:dims] = r.x
            if dims == 2:
                out[2] = float(test_df["rx_z"].iloc[0])
            preds.append(out)
        except Exception:
            preds.append([np.nan, np.nan, np.nan])
    return np.asarray(preds)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def error_stats(err: np.ndarray) -> dict:
    err = err[np.isfinite(err)]
    if err.size == 0:
        return dict(n_test=0, mean_err_m=np.nan, median_err_m=np.nan,
                    p90_err_m=np.nan, p95_err_m=np.nan, rmse_m=np.nan)
    return dict(
        n_test=int(err.size),
        mean_err_m=float(err.mean()),
        median_err_m=float(np.median(err)),
        p90_err_m=float(np.percentile(err, 90)),
        p95_err_m=float(np.percentile(err, 95)),
        rmse_m=float(np.sqrt(np.mean(err ** 2))),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="links.csv")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["within", "cross", "pooled"], default="within",
                    help="within=场景内按位置划分; cross=留一场景; pooled=场景内随机划分")
    ap.add_argument("--test-scenes", default=None,
                    help="逗号分隔的测试场景编号（可选，cross 模式下作为测试场景集合）")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standardize", choices=["none", "train"], default="train",
                    help="特征标准化方式，train 表示用训练集统计量")
    ap.add_argument("--baselines", nargs="*",
                    default=["knn_rss", "knn_rss_tof", "knn_rss_mp", "tof_trilat"])
    ap.add_argument("--n-anchors", type=int, default=None,
                    help="只保留锚点数等于该值的场景；默认取出现最多的锚点数")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    df = df[np.isfinite(df["rss_db"])]
    if df.empty:
        print("输入为空", file=sys.stderr)
        return 1

    # 锚点数对齐：默认取出现最多的锚点数
    per_scene_anchor = df.groupby("scene_id")["tx_id"].nunique()
    n_anchors = a.n_anchors or int(per_scene_anchor.mode().iloc[0])
    kept = per_scene_anchor[per_scene_anchor == n_anchors].index.tolist()
    dropped = [s for s in per_scene_anchor.index if s not in kept]
    print(f"锚点数对齐：保留 {len(kept)} 个场景（锚点数={n_anchors}）")
    if dropped:
        print(f"           丢弃 {len(dropped)} 个锚点数不匹配的场景：{dropped[:6]}{' ...' if len(dropped) > 6 else ''}")
    df = df[df["scene_id"].isin(kept)]

    scenes = sorted(df["scene_id"].unique())
    rng = np.random.default_rng(a.seed)
    outdir = Path(a.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for name in a.baselines:
        if name == "tof_trilat":
            # 几何定位：逐场景评估，无需训练集
            for scene_id in scenes:
                g = df[df["scene_id"] == scene_id]
                anchors = g.drop_duplicates("tx_id")[["tx_id", "tx_x", "tx_y", "tx_z"]]
                truth = g.drop_duplicates("rx_id")[["rx_id", "rx_x", "rx_y", "rx_z"]].sort_values("rx_id")
                pred = tof_trilaterate(g, anchors)
                err = np.linalg.norm(pred - truth[["rx_x", "rx_y", "rx_z"]].to_numpy(), axis=1)
                st = error_stats(err)
                st.update(baseline=name, mode="self", scene_id=scene_id, k=0,
                          train_scenes="", feature="tof")
                metric_rows.append(st)
                pred_rows.append(pd.DataFrame({
                    "baseline": name, "mode": "self", "scene_id": scene_id,
                    "rx_id": truth["rx_id"].to_numpy(),
                    "true_x": truth["rx_x"].to_numpy(), "true_y": truth["rx_y"].to_numpy(),
                    "true_z": truth["rx_z"].to_numpy(),
                    "pred_x": pred[:, 0], "pred_y": pred[:, 1], "pred_z": pred[:, 2],
                    "error_m": err,
                }))
            continue

        # ---- KNN 基线 ----
        feats = FEATURE_SETS[name.replace("knn_", "")]
        X, y, meta = build_fingerprints(df, feats, n_anchors=n_anchors)
        if X.size == 0:
            print(f"  {name}: 特征构造失败，跳过")
            continue
        scene_of = meta["scene_id"].to_numpy()

        # 按位置（不是按链路）划分，避免同一位置的信息泄漏到测试集
        if a.mode == "cross":
            test_scenes = (set(a.test_scenes.split(",")) if a.test_scenes else {scenes[-1]})
            folds = [(np.array([s != sc for s in scene_of]), np.array([s == sc for s in scene_of]), sc)
                     for sc in scenes if sc in test_scenes]
        else:
            folds = []
            for sc in scenes:
                m = scene_of == sc
                n = int(m.sum())
                perm = rng.permutation(n)
                n_tr = int(n * a.train_ratio)
                tr_local, te_local = perm[:n_tr], perm[n_tr:]
                tr = np.zeros(len(scene_of), dtype=bool)
                te = np.zeros(len(scene_of), dtype=bool)
                tr[np.flatnonzero(m)[tr_local]] = True
                te[np.flatnonzero(m)[te_local]] = True
                folds.append((tr, te, sc))

        for tr, te, sc in folds:
            if tr.sum() < 2 or te.sum() < 1:
                continue
            model = KnnFingerprint(k=a.k, standardize=a.standardize).fit(X[tr], y[tr])
            pred = model.predict(X[te])
            err = np.linalg.norm(pred - y[te], axis=1)
            st = error_stats(err)
            st.update(baseline=name, mode=a.mode, scene_id=sc, k=a.k,
                      train_scenes=str(int(tr.sum())), feature="+".join(feats))
            metric_rows.append(st)
            pred_rows.append(pd.DataFrame({
                "baseline": name, "mode": a.mode, "scene_id": scene_of[te],
                "rx_id": meta["rx_id"].to_numpy()[te],
                "true_x": y[te, 0], "true_y": y[te, 1], "true_z": y[te, 2],
                "pred_x": pred[:, 0], "pred_y": pred[:, 1], "pred_z": pred[:, 2],
                "error_m": err,
            }))
        print(f"  {name:14s} 已评估 {len(folds)} 折")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outdir / "metrics.csv", index=False, float_format="%.4f")
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            outdir / "predictions.csv", index=False, float_format="%.4f")

    print("\n== 汇总（按基线聚合全部折） ==")
    summary = (metrics.groupby("baseline")
               .agg(n_test=("n_test", "sum"),
                    mean_err=("mean_err_m", "mean"),
                    median=("median_err_m", "median"),
                    p90=("p90_err_m", "mean"),
                    p95=("p95_err_m", "mean"))
               .sort_values("median"))
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(summary)
    print(f"\n结果已写出：{outdir/'metrics.csv'}、{outdir/'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
