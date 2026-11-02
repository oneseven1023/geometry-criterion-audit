#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sensitivity_scan.py -- 判据侧动态范围扫描与角分辨率到阵列孔径的换算

一 动态范围 D 扫描
------------------
D 是判据侧参与构造位置 FIM 的路径功率门限（以组内最大功率为基准，10 log10）。
正文默认 30 dB。本脚本在 20、30、40 dB 下重算四个主指标，检验结论对 D 的依赖。

二 角分辨率到阵列孔径
----------------------
正文真实侧用 20 度作为角分辨率门限。该门限的物理来源是阵列孔径：均匀线阵的
半功率波束宽度约 0.886*lambda/D（D 为孔径），故给定角分辨率 theta 所需孔径为
D = 0.886*lambda/theta。脚本按若干候选频段给出孔径与半波间距阵元数。

输出 results/sensitivity_dr.csv；孔径换算直接打印。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio          # noqa: E402
import c2_fullscale_vec as c2v    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"
RES = Path(__file__).resolve().parent.parent / "results"

SPLITS = [("WiPTR", "val_grid"), ("WiPTR", "val_ood"), ("WiPTR", "train_grid")]
N_FLOOR = 20
DRS = [20.0, 30.0, 40.0]
C_M_PER_S = 299792458.0


def scan() -> pd.DataFrame:
    rows = []
    for ds, sp in SPLITS:
        info = next((s for s in wio.discover_splits(ROOT)
                     if s.tag == f"{ds}/{sp}"), None)
        if info is None:
            continue
        with h5py.File(info.path, "r") as f:
            floors = wio.list_floors(f)
            step = max(1, len(floors) // N_FLOOR)
            sel = floors[::step][:N_FLOOR]
            for dr in DRS:
                sz, ka, de, rk, uz = [], [], [], [], []
                for fid, _ in sel:
                    r = c2v.process_floor(f, fid, dr)
                    if r is None:
                        continue
                    sz.append(r["share_z"]); ka.append(r["kappa"])
                    de.append(r["d_eff"]); rk.append(r["rank"])
                    uz.append(r["mean_abs_uz"])
                if not sz:
                    continue
                sz = np.concatenate(sz); ka = np.concatenate(ka)
                de = np.concatenate(de); rk = np.concatenate(rk)
                uz = np.concatenate(uz)
                rows.append(dict(
                    split=f"{ds}/{sp}", dr_db=dr, n_floor=len(sel), n_link=sz.size,
                    rank3=float((rk == 3).mean()),
                    kappa_med=float(np.median(ka)),
                    deff_med=float(np.median(de)),
                    sharez_med=float(np.median(sz)),
                    sharez_gt05=float((sz > 0.5).mean()),
                    strict=float(((rk == 3) & (ka < 100) & (sz > 0.5)).mean()),
                    uz_med=float(np.median(uz)),
                ))
        print(f"[{ds}/{sp}] 完成 {len(sel)} 层 x {len(DRS)} 档")
    return pd.DataFrame(rows)


def aperture_table() -> pd.DataFrame:
    theta = np.radians(20.0)
    rows = []
    for fghz in (2.4, 3.5, 5.0, 6.0):
        lam = C_M_PER_S / (fghz * 1e9)
        D = 0.886 * lam / theta
        n_elem = int(np.ceil(D / (lam / 2.0))) + 1
        rows.append(dict(freq_ghz=fghz, lambda_m=lam, aperture_m=D,
                         n_halfwave_elements=n_elem))
    return pd.DataFrame(rows)


def main() -> int:
    df = scan()
    df.to_csv(RES / "sensitivity_dr.csv", index=False, float_format="%.4f")
    print("\n== 动态范围敏感性 ==")
    with pd.option_context("display.width", 150, "display.float_format",
                           "{:.4f}".format):
        print(df.to_string(index=False))

    ap = aperture_table()
    ap.to_csv(RES / "sensitivity_aperture.csv", index=False, float_format="%.4f")
    print("\n== 20 度角分辨率对应的阵列孔径（半功率波束宽度 0.886*lambda/D） ==")
    with pd.option_context("display.width", 150, "display.float_format",
                           "{:.4f}".format):
        print(ap.to_string(index=False))
    print(f"\n写出 {RES / 'sensitivity_dr.csv'} 与 "
          f"{RES / 'sensitivity_aperture.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
