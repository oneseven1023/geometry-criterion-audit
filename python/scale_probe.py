#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scale_probe.py -- 统计每个 split 的楼层数、信道数、按真实坐标分组后的接收机数。

用于确定全量审计的规模与耗时，并给出「按 rx 真实坐标分组」后的实际样本数
（这才是审计脚本的真正样本单位）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiinsim_io as wio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "public_raw"


def main():
    tot_ch = 0
    tot_rx = 0
    print(f"{'split':<32}{'floors':>7}{'channels':>12}{'rx(unique)':>12}{'tx':>6}")
    print("-" * 72)
    for info in wio.discover_splits(ROOT):
        with h5py.File(info.path, "r") as f:
            fl = wio.list_floors(f)
            n_ch = int(np.asarray(f["channels"].shape[0]))
            n_tx = int(np.asarray(f["tx"].shape[0]))
            n_rx_glob = int(np.asarray(f["rx"].shape[0]))
            # 按真实坐标分组的接收机数（全部楼层累计）
            n_rx = 0
            for fid, _ in fl:
                fc = wio.read_floor(f, fid)
                n_rx += int(np.unique(fc.rx_xyz, axis=0).shape[0])
        print(f"{info.tag:<32}{len(fl):>7}{n_ch:>12}{n_rx:>12}{n_tx:>6}"
              f"   (文件内 rx 行数 {n_rx_glob})")
        tot_ch += n_ch
        tot_rx += n_rx
    print("-" * 72)
    print(f"合计：信道 {tot_ch}，分组后接收机（链路组）{tot_rx}")


if __name__ == "__main__":
    main()
