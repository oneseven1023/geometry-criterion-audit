#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stratified_audit.py -- 统计单位与指标关系的分层审计

回应审稿意见第五条（数据口径）与 P1-1（统计独立性）：

1. 逐划分的锚点数分布，回答「Wi3R/main_5tx 是否为单锚点链路」。
2. 把核心比例按「单锚点 / 多锚点」「Wi3R / WiPTR」分层重算，检验结论是否由
   某一类样本独撑。
3. 以 (划分, 楼层) 为簇做聚类自助，给出核心比例的置信区间（比按链路独立更宽，
   比按划分重采样更细）。
4. 计算条件数与垂直方差份额、平均到达方向垂直分量在各划分内的秩相关，用于
   检验「条件数是否稳定指示垂直退化」。

输出
----
results/stratified_audit.csv   分层比例表
results/ratio_ci_floor.csv     楼层级聚类自助区间
results/metric_relation.csv    条件数与方向量的秩相关（逐划分）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ratios(d: pd.DataFrame) -> dict:
    k = d["kappa"].to_numpy()
    s = d["share_z"].to_numpy()
    return dict(
        n=len(d),
        rank3=float((d["rank"].to_numpy() == 3).mean()),
        kappa_lt100=float((k < 100).mean()),
        deff_lt2=float((d["d_eff"].to_numpy() < 2).mean()),
        sharez_gt05=float((s > 0.5).mean()),
        strict_fp=float(((d["rank"].to_numpy() == 3) & (k < 100) & (s > 0.5)).mean()),
        kappa_med=float(np.median(k)),
        deff_med=float(np.median(d["d_eff"].to_numpy())),
        sharez_med=float(np.median(s)),
        mean_uz_med=float(np.median(d["mean_abs_uz"].to_numpy())),
    )


def main() -> int:
    frames = []
    for f in sorted(RES.glob("c2full_*.csv")):
        tag = f.name.replace("c2full_", "").replace(".csv", "")
        d = pd.read_csv(f, usecols=["n_anchor", "rank", "kappa", "d_eff", "share_z",
                                    "mean_abs_uz", "floor"])
        d["split"] = tag
        d["dataset"] = tag.split("_")[0]
        frames.append(d)
    A = pd.concat(frames, ignore_index=True)

    # ---- 1/2 分层比例 ----
    groups = [("全部", A),
              ("单锚点 (n=1)", A[A.n_anchor == 1]),
              ("多锚点 (n>=2)", A[A.n_anchor >= 2]),
              ("Wi3R 全部", A[A.dataset == "Wi3R"]),
              ("WiPTR 全部", A[A.dataset == "WiPTR"]),
              ("Wi3R 多锚点", A[(A.dataset == "Wi3R") & (A.n_anchor >= 2)]),
              ("WiPTR 多锚点", A[(A.dataset == "WiPTR") & (A.n_anchor >= 2)])]
    rows = [dict(group=g, **ratios(d)) for g, d in groups]
    pd.DataFrame(rows).to_csv(RES / "stratified_audit.csv", index=False,
                              float_format="%.4f")

    # ---- 逐划分锚点数分布 ----
    print("=== 逐划分锚点数分布 ===")
    for tag, d in A.groupby("split"):
        vc = d["n_anchor"].value_counts().sort_index()
        print(f"  {tag:34s} n={len(d):8d}  {dict(vc)}")

    print("\n=== 分层比例 ===")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 3 楼层级聚类自助 ----
    cl = (A.groupby(["split", "floor"])
            .agg(n=("rank", "size"),
                 rank3=("rank", lambda x: int((x == 3).sum())),
                 shz=("share_z", lambda x: int((x > 0.5).sum())))
            .reset_index())
    kc = A.assign(k100=(A.kappa < 100).astype(int),
                  strict=((A["rank"] == 3) & (A.kappa < 100) & (A.share_z > 0.5)).astype(int))
    cl = cl.merge(kc.groupby(["split", "floor"]).agg(
        k100=("k100", "sum"), strict=("strict", "sum")).reset_index(),
        on=["split", "floor"])
    X = {c: cl[c].to_numpy(float) for c in ("n", "rank3", "shz", "k100", "strict")}
    B = 2000
    rng = np.random.default_rng(20261012)
    m = cl.shape[0]
    boot = {c: np.empty(B) for c in ("rank3", "shz", "k100", "strict")}
    for b in range(B):
        idx = rng.integers(0, m, m)
        nn = X["n"][idx].sum()
        boot["rank3"][b] = X["rank3"][idx].sum() / nn
        boot["shz"][b] = X["shz"][idx].sum() / nn
        boot["k100"][b] = X["k100"][idx].sum() / nn
        boot["strict"][b] = X["strict"][idx].sum() / nn
    pool = dict(rank3=float((A["rank"] == 3).mean()),
                shz=float((A.share_z > 0.5).mean()),
                k100=float((A.kappa < 100).mean()),
                strict=float(((A["rank"] == 3) & (A.kappa < 100)
                              & (A.share_z > 0.5)).mean()))
    ci = []
    for c in ("rank3", "shz", "k100", "strict"):
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        ci.append(dict(metric=c, pool=pool[c], ci_lo=float(lo), ci_hi=float(hi),
                       n_floor=int(m)))
    pd.DataFrame(ci).to_csv(RES / "ratio_ci_floor.csv", index=False, float_format="%.4f")
    print(f"\n=== 楼层级聚类自助（簇 {m} 个，B={B}） ===")
    print(pd.DataFrame(ci).to_string(index=False))

    # ---- 4 条件数与方向量的秩相关 ----
    rel = []
    for tag, d in A.groupby("split"):
        rel.append(dict(split=tag, n=len(d),
                        rankcorr_kappa_sharez=rank_corr(d.kappa.to_numpy(),
                                                        d.share_z.to_numpy()),
                        rankcorr_kappa_meanuz=rank_corr(d.kappa.to_numpy(),
                                                        d.mean_abs_uz.to_numpy())))
    rel.append(dict(split="Wi3R 合计", n=int((A.dataset == "Wi3R").sum()),
                    rankcorr_kappa_sharez=rank_corr(A[A.dataset == "Wi3R"].kappa.to_numpy(),
                                                    A[A.dataset == "Wi3R"].share_z.to_numpy()),
                    rankcorr_kappa_meanuz=rank_corr(A[A.dataset == "Wi3R"].kappa.to_numpy(),
                                                    A[A.dataset == "Wi3R"].mean_abs_uz.to_numpy())))
    rel.append(dict(split="WiPTR 合计", n=int((A.dataset == "WiPTR").sum()),
                    rankcorr_kappa_sharez=rank_corr(A[A.dataset == "WiPTR"].kappa.to_numpy(),
                                                    A[A.dataset == "WiPTR"].share_z.to_numpy()),
                    rankcorr_kappa_meanuz=rank_corr(A[A.dataset == "WiPTR"].kappa.to_numpy(),
                                                    A[A.dataset == "WiPTR"].mean_abs_uz.to_numpy())))
    R = pd.DataFrame(rel)
    R.to_csv(RES / "metric_relation.csv", index=False, float_format="%.4f")
    print("\n=== 条件数与方向量的秩相关 ===")
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
