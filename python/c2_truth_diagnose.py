#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_truth_diagnose.py -- 诊断扩样中出现的「真值以外极小点」是真歧义还是数值停滞

背景
----
`c2_truth_stratified.py` 在 5303 条分层样本上发现 87 条（1.64%）链路的多起点求解
出现真值以外的极小点，试点 400 条时为 0。特征指向单锚点（n_anchor=1）：
64 个初始点中约 18 个收敛到别处，远解 z 偏移中位约 1.8 m。

本脚本对这批链路做细粒度诊断，区分两种可能：
  A 真歧义：虚拟锚点构型存在离散简并（如近共面导致关于拟合平面的镜像解）
  B 数值停滞：近奇异方向上的平坦谷，远解残差显著大于真值残差

判别量
------
1. 每簇解的无噪声残差 rms（真解应 ~0；若远解残差亦 ~1e-3 m 量级，则为真极小）
2. 虚拟锚点 VA 点云的 PCA 奇异值比 s3/s1（近 0 表示共面，支持镜像解解释）
3. 把 x_true 关于 VA 最佳拟合平面做镜像，比较镜像点与远解的距离

用法
----
python c2_truth_diagnose.py --csv results/c2truth_all.csv --max 60
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402
from c2_independent_truth import gn_range_solve  # noqa: E402
from c2_truth_stratified import floor_links  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
C_M_PER_NS = wio.C_M_PER_NS


def residual_rms(X, VA, r, sig):
    d = np.linalg.norm(X[None, :] - VA, axis=1)
    return float(np.sqrt(np.mean(((d - r) / sig) ** 2)))


def cluster_solutions(X: np.ndarray, tol: float = 0.30):
    """把解按 tol 聚簇，返回 [(代表点, 成员数)]。"""
    centres, counts = [], []
    for p in X:
        for i, c in enumerate(centres):
            if np.linalg.norm(p - c) < tol:
                centres[i] = (c * counts[i] + p) / (counts[i] + 1)
                counts[i] += 1
                break
        else:
            centres.append(p.copy())
            counts.append(1)
    return list(zip(centres, counts))


def diagnose_one(L, gi, x_true, n_init=256, seed=1234):
    s = L["st2"][gi]
    e = s + L["cnt2"][gi]
    dirs, w, tau = L["d_o"][s:e], L["w_o"][s:e], L["tau_o"][s:e]
    VA = x_true[None, :] - C_M_PER_NS * tau[:, None] * dirs
    K = VA.shape[0]
    dist_true = np.linalg.norm(x_true[None, :] - VA, axis=1)

    rng = np.random.default_rng(seed)
    X0 = x_true[None, :] + rng.normal(0.0, 3.0, size=(n_init, 3))
    r_b = np.broadcast_to(dist_true, (n_init, K))
    Xh = gn_range_solve(VA, r_b, np.ones(K), X0, iters=400, damp=1e-7)

    cl = cluster_solutions(Xh)
    # 每簇残差与位置
    info = []
    for c, cnt in cl:
        info.append(dict(centre=c, count=int(cnt),
                         resid=residual_rms(c, VA, dist_true, np.ones(K)),
                         dist_true=float(np.linalg.norm(c - x_true))))
    info.sort(key=lambda t: -t["count"])

    # VA 点云共面性
    Vc = VA - VA.mean(axis=0)
    sv = np.linalg.svd(Vc, compute_uv=False)
    sv_ratio = float(sv[2] / sv[0]) if sv[0] > 0 else 0.0

    # 最佳拟合平面的镜像解
    _, _, Vt = np.linalg.svd(Vc)
    n_hat = Vt[2]
    c0 = VA.mean(axis=0)
    mir = x_true - 2.0 * ((x_true - c0) @ n_hat) * n_hat
    mir_resid = residual_rms(mir, VA, dist_true, np.ones(K))

    # 远解是否接近镜像点
    others = [t for t in info if t["dist_true"] > 0.25]
    far = others[0] if others else None
    d_to_mirror = float(np.linalg.norm(far["centre"] - mir)) if far else float("nan")

    return dict(n_sol=len(info), n_other_clusters=len(others),
                sv_ratio=sv_ratio,
                resid_true=residual_rms(x_true, VA, dist_true, np.ones(K)),
                far=far, mir_resid=mir_resid, d_to_mirror=d_to_mirror)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent.parent
                                         / "results" / "c2truth_all.csv"))
    ap.add_argument("--max", type=int, default=60)
    args = ap.parse_args()

    import csv as _csv
    with open(args.csv, encoding="utf-8") as fh:
        rows = [r for r in _csv.DictReader(fh) if int(float(r["ambig"])) == 1]
    rows.sort(key=lambda r: -int(float(r["n_other"])))
    rows = rows[:args.max]
    print(f"诊断歧义链路 {len(rows)} 条（按 n_other 降序）")

    splits = {s.tag: s for s in wio.discover_splits(ROOT)}
    by_floor = defaultdict(list)
    for r in rows:
        by_floor[(r["split"], int(float(r["floor_id"])))].append(r)

    agg = []
    print(f"\n{'split':<24}{'flo':>6}{'n_p':>5}{'nva':>5}{'s3/s1':>8}"
          f"{'r_tru':>9}{'r_far':>9}{'dz_far':>8}{'r_mir':>9}{'d->mir':>8}")
    print("-" * 96)
    for (tag, fid), rs in by_floor.items():
        info = splits[tag]
        with h5py.File(info.path, "r") as f:
            L = floor_links(f, fid, 30.0)
            if L is None:
                continue
            for r in rs:
                xt = np.array([float(r["rx_x"]), float(r["rx_y"]), float(r["rx_z"])])
                # 找对应组：坐标严格相等
                d = np.linalg.norm(L["rx_xyz_g"] - xt[None, :], axis=1)
                gi = int(np.argmin(d))
                if d[gi] > 1e-6:
                    continue
                res = diagnose_one(L, gi, xt)
                far = res["far"]
                print(f"{tag:<24}{fid:>6}{L['n_path'][gi]:>5}{int(r['n_anchor']):>5}"
                      f"{res['sv_ratio']:>8.4f}{res['resid_true']:>9.2e}"
                      f"{(far['resid'] if far else float('nan')):>9.2e}"
                      f"{(far['centre'][2]-xt[2] if far else float('nan')):>8.3f}"
                      f"{res['mir_resid']:>9.2e}{res['d_to_mirror']:>8.3f}")
                agg.append(dict(sv=res["sv_ratio"], r_true=res["resid_true"],
                                r_far=(far["resid"] if far else np.nan),
                                d_mir=res["d_to_mirror"],
                                r_mir=res["mir_resid"],
                                n_cl=res["n_other_clusters"],
                                n_anchor=int(r["n_anchor"]), n_path=int(L["n_path"][gi])))
    if not agg:
        print("无有效诊断样本"); return
    sv = np.array([a["sv"] for a in agg])
    rt = np.array([a["r_true"] for a in agg])
    rf = np.array([a["r_far"] for a in agg])
    dm = np.array([a["d_mir"] for a in agg])
    rm = np.array([a["r_mir"] for a in agg])
    print(f"\n汇总（{len(agg)} 条）：")
    print(f"  VA 点云 s3/s1 中位 {np.median(sv):.4f}  （越小越接近共面）")
    print(f"  真值残差 rms 中位 {np.median(rt):.2e}")
    print(f"  远解残差 rms 中位 {np.nanmedian(rf):.2e}")
    print(f"  远解残差 / 真值残差 中位 {np.nanmedian(rf/rt):.2f}")
    print(f"  镜像解残差 rms 中位 {np.median(rm):.2e}")
    print(f"  远解到镜像点距离 中位 {np.nanmedian(dm):.3f} m")
    print(f"  远解≈镜像(距离<0.5m) 的比例 {np.nanmean(dm<0.5):.3f}")
    print(f"  远解残差与真值残差同量级(比值<10) 的比例 {np.nanmean(rf/rt<10):.3f}")


if __name__ == "__main__":
    main()
