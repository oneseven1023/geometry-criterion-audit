#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""validate_audit_tool.py -- 校验审计工具的两条口径一致性

做两件事：

1. **对全量脚本**：把 ``identifiability_audit.py`` 的 h5 后端在某个划分上的
   指标中位与 ``results/c2full_<split>.csv``（由 ``c2_fullscale_vec.py`` 产出）
   对齐，证明工具复现了论文全量口径。
2. **跨后端**：把归一化 CSV 后端与 h5 后端放在**同一批楼层**上比较。前提是
   links.csv 与 paths.csv 覆盖同一批信道；若快照内的 paths 是部分导出，
   本脚本会先生成一份一致的快照（``--rebuild``）再比较。

用法
----
    # 只做第 1 项
    python python/validate_audit_tool.py --split WiPTR/val_grid

    # 第 1 项加第 2 项（会临时调用 normalize_rt.py 生成一致快照）
    python python/validate_audit_tool.py --split WiPTR/val_grid \
        --rebuild-splits train_grid val_grid --rebuild-floors 2
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identifiability_audit as IA  # noqa: E402
import wiinsim_io as wio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["rank", "kappa", "d_eff", "share_z", "mean_abs_uz"]


def check_against_fullscale(split_tag: str, dr: float) -> bool:
    """第 1 项：与 c2_fullscale_vec 的全量 CSV 对齐。"""
    csv_name = "c2full_" + split_tag.replace("/", "_") + ".csv"
    ref_csv = ROOT / "results" / csv_name
    if not ref_csv.exists():
        print(f"[跳过] 未找到 {ref_csv.name}，无法做全量对齐")
        return False

    d, rt = IA.audit_h5(ROOT / "data" / "public_raw", split_tag, dr)
    with open(ref_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ref = {k: np.asarray([float(r[k]) for r in rows]) for k in METRICS}

    print(f"=== 对全量脚本（{split_tag}，链路 {d['rank'].shape[0]:,}）===")
    ok = True
    for k in METRICS:
        a, b = float(np.median(d[k].astype(float))), float(np.median(ref[k]))
        rel = abs(a - b) / (abs(b) if b else 1.0)
        flag = "OK " if rel < 1e-9 else "!! "
        ok &= rel < 1e-9
        print(f"  {flag}{k:12s} 工具 {a:14.8f} | 全量 {b:14.8f} | 相对偏差 {rel:.2e}")
    print(f"  用时 {rt:.2f}s")
    return ok


def rebuild_snapshot(splits: list[str], floors: int, outdir: Path) -> None:
    """调用 normalize_rt.py 生成 links 与 paths 覆盖同一批楼层的快照。"""
    cmd = [sys.executable, str(ROOT / "python" / "normalize_rt.py"),
           "--source", "public",
           "--public-root", str(ROOT / "data" / "public_raw"),
           "--outdir", str(outdir),
           "--datasets", "WiPTR",
           "--splits", *splits,
           "--max-floors-per-split", str(floors),
           "--with-paths", "--max-path-floors", str(floors)]
    print("[rebuild]", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=ROOT)


def check_across_backends(snap: Path, dr: float) -> bool:
    """第 2 项：同一批楼层上 CSV 后端对 h5 后端。"""
    import pandas as pd

    links = pd.read_csv(snap / "links.csv")
    splits = {s.tag: s for s in wio.discover_splits(ROOT / "data" / "public_raw")}

    dn, tn = IA.audit_normalized(snap / "links.csv", snap / "paths.csv", dr)

    acc = {k: [] for k in IA.LINK_KEYS}
    for (ds, sp), sub in links.groupby(["dataset", "split"]):
        info = splits[f"{ds}/{sp}"]
        with wio.h5py.File(info.path, "r") as f:
            idx2row = {idx: fid for fid, idx in wio.list_floors(f)}
            for idx in sorted(set(int(x) for x in sub.floor_idx)):
                r = IA._floor_metrics(f, idx2row[idx], dr)
                if r is None:
                    continue
                n = r["rank"].shape[0]
                acc["group"].append(np.full(n, f"{ds}/{sp}", dtype=object))
                for k in IA.LINK_KEYS[1:]:
                    acc[k].append(r[k])
    dh = IA._concat(acc)

    print(f"\n=== 跨后端（同一批楼层，CSV {dn['rank'].shape[0]:,} 对 "
          f"h5 {dh['rank'].shape[0]:,} 链路）===")
    if dn["rank"].shape[0] != dh["rank"].shape[0]:
        print("  !! 链路数不一致，快照与 h5 不同源，请加 --rebuild")
    ok = dn["rank"].shape[0] == dh["rank"].shape[0]
    for k in METRICS + ["n_anchor", "n_path"]:
        a, b = float(np.median(dn[k].astype(float))), float(np.median(dh[k].astype(float)))
        rel = abs(a - b) / (abs(b) if b else 1.0)
        flag = "OK " if rel < 1e-4 else "!! "
        ok &= rel < 1e-4
        print(f"  {flag}{k:12s} CSV {a:14.8f} | h5 {b:14.8f} | 相对偏差 {rel:.2e}")
    a = dn["strict_false_pass"].mean() * 100
    b = dh["strict_false_pass"].mean() * 100
    ok &= abs(a - b) < 1e-6
    print(f"  OK 严格假通过%  CSV {a:14.6f} | h5 {b:14.6f}")
    print(f"  CSV 后端用时 {tn:.2f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="WiPTR/val_grid")
    ap.add_argument("--dr", type=float, default=30.0)
    ap.add_argument("--rebuild-splits", nargs="*", default=None)
    ap.add_argument("--rebuild-floors", type=int, default=2)
    ap.add_argument("--snapshot-dir",
                    default=str(ROOT / "results" / "audit_validation_snapshot"))
    args = ap.parse_args()

    ok1 = check_against_fullscale(args.split, args.dr)

    ok2 = None
    if args.rebuild_splits:
        snap = Path(args.snapshot_dir)
        rebuild_snapshot(args.rebuild_splits, args.rebuild_floors, snap)
        ok2 = check_across_backends(snap, args.dr)

    print("\n结论：", "全部通过" if (ok1 and ok2 is not False) else "存在未通过项")
    return 0 if (ok1 and ok2 is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
