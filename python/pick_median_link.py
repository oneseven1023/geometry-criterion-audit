"""为图 1(b) 选取一条中位代表链路，规则客观、不挂钩结论。

选取规则（写进论文图注，可复现）：
  数据源 WiPTR/val_grid 的第 0 层，按 (rx, tx) 组合分组得到全部链路，
  丢弃路径数少于 15 条的链路，然后取垂直方差份额 s_z 最接近
  该层全体链路 s_z 中位数的那一条。

与旧规则的区别：旧规则取 s_z 最接近 0.90 的链路，0.90 远高于全量中位数
0.5905，属按结论挑选，不具代表性。新规则以中位数为目标，选取不依赖结论。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h5py
import wiinsim_io as wio
from c2_criterion_audit import fim, fim_metrics, path_directions, path_power

ROOT = HERE.parent
OUT = HERE.parent / "figures_build" / "fig1_geometry_virtual_anchor" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    info = {s.tag: s for s in wio.discover_splits(ROOT / "data" / "public_raw")}
    target = info["WiPTR/val_grid"]
    with h5py.File(target.path, "r") as f:
        fid = [i for i, _ in wio.list_floors(f)][0]
        fc = wio.read_floor(f, fid)
        paths, ch_of = wio.read_floor_paths(f, fid)
        c0, c1 = int(wio.floor_channel_bounds(f)[fid]), int(wio.floor_channel_bounds(f)[fid + 1])
        ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)

    dirs = path_directions(paths)
    w = path_power(paths)
    _, uid = np.unique(fc.rx_xyz, axis=0, return_inverse=True)
    tx_of_channel = ch[:, 0]
    n_tx = int(tx_of_channel.max()) + 1
    g = uid[ch_of].astype(np.int64) * n_tx + tx_of_channel[ch_of]

    rows = []
    for gi in np.unique(g):
        sel = g == gi
        if sel.sum() < 15:
            continue
        ww = w[sel] / w[sel].max()
        m = fim_metrics(fim(dirs[sel], ww))
        rows.append(
            dict(gi=int(gi), n=int(sel.sum()), sel=sel, dirs=dirs[sel], w=ww,
                 share_z=float(m["share_z"]), kappa=float(m["kappa"]),
                 d_eff=float(m["d_eff"]), lam=np.asarray(m["lam"], dtype=float))
        )

    sz = np.array([r["share_z"] for r in rows])
    med_layer = float(np.median(sz))
    print(f"本层链路数 (>=15 paths): {len(rows)}")
    print(f"本层 s_z 中位数        : {med_layer:.4f}")
    print(f"本层 s_z 四分位        : {np.percentile(sz, 25):.4f} / "
          f"{np.percentile(sz, 75):.4f}")
    print(f"本层 s_z 均值          : {sz.mean():.4f}")
    print(f"旧图所用链路 s_z       : 0.9000 (本层第 "
          f"{100.0 * (sz < 0.90).mean():.1f} 百分位)")

    # 中位数意义上的代表链路
    best = min(rows, key=lambda r: abs(r["share_z"] - med_layer))
    print()
    print(f"选中链路: n_path={best['n']}  s_z={best['share_z']:.4f}  "
          f"kappa={best['kappa']:.1f}  d_eff={best['d_eff']:.3f}  "
          f"lam={best['lam']}")
    print(f"其 s_z 在本层的百分位  : {100.0 * (sz < best['share_z']).mean():.1f}%")
    mean_abs_uz = float(np.abs(best["dirs"][:, 2]).mean())
    print(f"mean|u_z|              : {mean_abs_uz:.4f}")
    uz = best["dirs"][:, 2]
    print(f"|u_z|<1e-6 的条数       : {int((np.abs(uz) < 1e-6).sum())} / {best['n']}")
    nz = np.abs(uz[np.abs(uz) >= 1e-6])
    print(f"非零 |u_z| 范围         : {nz.min():.4f} 到 {nz.max():.4f}")

    import pandas as pd

    df = pd.DataFrame({
        "ux": best["dirs"][:, 0], "uy": best["dirs"][:, 1],
        "uz": best["dirs"][:, 2], "w_norm": best["w"],
    })
    out_csv = OUT / "fig1_representative_link.csv"
    df.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"\n写出 {out_csv}")

    stats = dict(
        split="WiPTR/val_grid", floor_id=0, n_path=int(best["n"]),
        d_eff=best["d_eff"], kappa=best["kappa"], share_z=best["share_z"],
        lam_min=float(best["lam"][0]), lam_mid=float(best["lam"][1]),
        lam_max=float(best["lam"][2]),
        mean_abs_uz=mean_abs_uz,
        selection="s_z nearest to the layer median of s_z",
        layer_n_link=len(rows), layer_median_share_z=med_layer,
        layer_q25_share_z=float(np.percentile(sz, 25)),
        layer_q75_share_z=float(np.percentile(sz, 75)),
        share_z_percentile=float(100.0 * (sz < best["share_z"]).mean()),
        n_zero_uz=int((np.abs(uz) < 1e-6).sum()),
        n_nonzero_uz=int((np.abs(uz) >= 1e-6).sum()),
        nonzero_uz_min=float(nz.min()), nonzero_uz_max=float(nz.max()),
    )
    out_json = OUT / "fig1_stats.json"
    out_json.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"写出 {out_json}")


if __name__ == "__main__":
    main()
