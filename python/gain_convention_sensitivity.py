# -*- coding: utf-8 -*-
"""增益换算约定的敏感性：单径线性增益除数取 5 与 10 时经验结论如何变化。

论文正文用 ``w = 10^(g/5)``（``wiinsim`` 官方口径，g 为半 dB 表示）。本脚本
在同一份归一化数据、同一套指标口径下只改 ``GAIN_DB_DIVISOR``，各约定内部
自洽地保留 30 dB 动态范围，从而回答「换一套权重会不会只改变尺度」这一问题。

结论（30,086 链路）：份额类量几乎不变，绝对尺度类量变化很大。
    share_z 中位 0.8022 -> 0.7998（-0.29%）
    kappa 中位   40.376 -> 16.264（-59.7%）
    d_eff 中位    1.472 -> 1.893 （+28.6%）
    严格假通过    0.6070 -> 0.7685（+26.6%）
因此「只改变绝对尺度」的说法不成立，正文据此按实测敏感性表述。

用法：项目根目录下运行
    python python/gain_convention_sensitivity.py
输出：stdout（不写盘，避免污染 results/）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import identifiability_audit as A   # noqa: E402
import wiinsim_io as wio            # noqa: E402

LINKS = ROOT / "data" / "normalized" / "links.csv"
PATHS = ROOT / "data" / "normalized" / "paths.csv"

KEYS = ("rank3_frac", "kappa_median", "d_eff_median", "d_eff_lt2_frac",
        "share_z_median", "share_z_gt0.5_frac", "strict_false_pass_frac",
        "corr_sharez_meanabsuz")


def main() -> int:
    res = {}
    for div in (5.0, 10.0):
        wio.GAIN_DB_DIVISOR = div
        t0 = time.time()
        d, secs = A.audit_normalized(LINKS, PATHS, 30.0)
        s = A.summarize(d, secs, f"normalized/divisor={div:g}", None)
        res[div] = s
        print(f"--- divisor {div:g}: {s['n_links']} links, {secs:.2f}s")
        for k in KEYS:
            v = s[k]
            print(f"      {k:26s} {v:.6f}" if isinstance(v, float) else f"      {k}: {v}")

    a, b = res[5.0], res[10.0]
    print()
    print("=== 两套约定的偏差 ===")
    for k in KEYS[:-1]:
        x, y = float(a[k]), float(b[k])
        rel = (y - x) / x if x else float("nan")
        print(f"  {k:26s} div5={x:.6f}  div10={y:.6f}  相对变化={rel * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
