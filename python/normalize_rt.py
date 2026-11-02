#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""normalize_rt.py -- 把公开 RT 数据与自研 RT 输出统一为 links.csv / paths.csv

统一后的两张核心表
------------------
``links.csv``  每行 = 一个接收机位置与一个锚点之间的一条信道
``paths.csv``  每行 = 一条 RT 路径（仅在 ``--with-paths`` 时输出）

两条数据来源
------------
1. ``--source public``   直接读取 Qualcomm Wi3R / WiPTR 的 dataset_sparse.h5
2. ``--source synthetic`` 读取自研 RT 引擎导出的原始 CSV，按 docs/rt_output_format.md
                          的字段映射转换为同一套列名

用法
----
    # 先做小样本冒烟测试（每划分最多 3 个场景）
    python python/normalize_rt.py --source public \
        --public-root data/public_raw --outdir data/normalized \
        --splits val_ood val_grid train_grid --max-floors-per-split 3

    # 全量转换指定划分
    python python/normalize_rt.py --source public \
        --public-root data/public_raw --outdir data/normalized \
        --datasets WiPTR --splits train val test

    # 自研 RT 输出
    python python/normalize_rt.py --source synthetic \
        --raw-dir data/synthetic/raw --outdir data/normalized

输出列说明见 docs/data_schema.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiinsim_io import (  # noqa: E402
    GAIN_DB_DIVISOR,
    PATH_COLUMNS,
    SplitInfo,
    discover_splits,
    floor_channel_bounds,
    list_floors,
    read_floor,
)

LINK_COLUMNS = [
    "scene_id", "dataset", "split", "floor_id", "floor_idx",
    "tx_id", "rx_id",
    "rx_x", "rx_y", "rx_z", "tx_x", "tx_y", "tx_z",
    "rss_db", "tof_ns", "tof_strongest_ns", "path_count", "delay_spread_ns",
    "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg",
    "los_flag", "d_geo_m",
]

PATH_COLUMNS_OUT = [
    "scene_id", "tx_id", "rx_id", "path_id",
    "delay_ns", "gain_re", "gain_im", "gain_db",
    "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg",
    "interaction_type", "interaction_xyz", "material_id",
]


def make_scene_id(dataset: str, split: str, floor_idx: int) -> str:
    """场景唯一编号。同一 floor_idx 可能在不同 split 下重复，故必须带 split。"""
    safe_split = split.replace("/", "-")
    return f"{dataset}-{safe_split}-{floor_idx:05d}"


def normalize_public(
    public_root: Path,
    outdir: Path,
    datasets: list[str] | None,
    splits: list[str] | None,
    max_floors_per_split: int | None,
    with_paths: bool,
    max_path_floors: int,
) -> None:
    infos = discover_splits(public_root)
    if datasets:
        infos = [i for i in infos if i.dataset in set(datasets)]
    if splits:
        infos = [i for i in infos if i.split in set(splits) or i.split.split("/")[-1] in set(splits)]
    if not infos:
        print("没有匹配的划分，请检查 --datasets / --splits", file=sys.stderr)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    links_path = outdir / "links.csv"
    paths_path = outdir / "paths.csv"
    meta_rows: list[dict] = []

    if links_path.exists():
        links_path.unlink()
    if with_paths and paths_path.exists():
        paths_path.unlink()

    first_links = True
    first_paths = True
    for info in infos:
        n_floors_written = 0
        with h5py.File(info.path, "r") as f:
            floors = list_floors(f)
            if max_floors_per_split is not None:
                floors = floors[:max_floors_per_split]
            for floor_id, floor_idx in floors:
                try:
                    d = read_floor(f, floor_id)
                except ValueError:
                    continue
                scene_id = make_scene_id(info.dataset, info.split, floor_idx)

                # ---- 链路表 ----
                df = pd.DataFrame({
                    "scene_id": scene_id,
                    "dataset": info.dataset,
                    "split": info.split,
                    "floor_id": d.floor_id,
                    "floor_idx": d.floor_idx,
                    "tx_id": d.tx_idx,
                    "rx_id": d.rx_pos_id,
                    "rx_x": d.rx_xyz[:, 0], "rx_y": d.rx_xyz[:, 1], "rx_z": d.rx_xyz[:, 2],
                    "tx_x": d.tx_xyz[:, 0], "tx_y": d.tx_xyz[:, 1], "tx_z": d.tx_xyz[:, 2],
                    "rss_db": d.rss_db,
                    "tof_ns": d.tof_first_ns,
                    "tof_strongest_ns": d.tof_strong_ns,
                    "path_count": d.path_count,
                    "delay_spread_ns": d.delay_spread_ns,
                    "aoa_az_deg": d.aoa_az_deg, "aoa_el_deg": d.aoa_el_deg,
                    "aod_az_deg": d.aod_az_deg, "aod_el_deg": d.aod_el_deg,
                    "los_flag": d.los_flag,
                    "d_geo_m": d.d_geo_m,
                })[LINK_COLUMNS]

                df.to_csv(links_path, mode="w" if first_links else "a",
                          header=first_links, index=False, float_format="%.4f")
                first_links = False

                # ---- 场景元数据 ----
                rx_span = d.rx_xyz[:, :2].max(axis=0) - d.rx_xyz[:, :2].min(axis=0)
                meta_rows.append(dict(
                    scene_id=scene_id, dataset=info.dataset, split=info.split,
                    floor_idx=d.floor_idx,
                    n_channel=len(d),
                    n_tx=int(np.unique(d.tx_idx).size),
                    n_rx_pos=int(np.unique(d.rx_pos_id).size),
                    rx_count_per_tx=int(np.bincount(d.tx_idx).max()),
                    los_ratio=float(d.los_flag.mean()),
                    rss_median_db=float(np.median(d.rss_db)),
                    rss_min_db=float(d.rss_db.min()),
                    rss_max_db=float(d.rss_db.max()),
                    rss_span_db=float(d.rss_db.max() - d.rss_db.min()),
                    ds_median_ns=float(np.median(d.delay_spread_ns)),
                    rx_extent_x_m=float(rx_span[0]), rx_extent_y_m=float(rx_span[1]),
                    rx_bbox_area_m2=float(rx_span[0] * rx_span[1]),
                ))

                # ---- 逐路径表（可选） ----
                if with_paths and n_floors_written < max_path_floors:
                    bounds = floor_channel_bounds(f)
                    c0, c1 = int(bounds[floor_id]), int(bounds[floor_id + 1])
                    ch = np.asarray(f["channels"][c0:c1], dtype=np.int64)
                    p0 = int(ch[0, 2])
                    p_end = (int(np.asarray(f["channels"][c1, 2]))
                             if c1 < np.asarray(f["channels"].shape[0])
                             else int(np.asarray(f["paths"].shape[0])))
                    raw = np.asarray(f["paths"][p0:p_end], dtype=np.float64)
                    rel = ch[:, 2] - p0
                    counts = np.diff(np.append(rel, raw.shape[0]))
                    ch_local = np.repeat(np.arange(ch.shape[0]), counts)
                    pdf = pd.DataFrame({
                        "scene_id": scene_id,
                        "tx_id": ch[ch_local, 0],
                        "rx_id": (ch[ch_local, 1] % int(np.bincount(ch[:, 0]).max())),
                        "path_id": np.concatenate([np.arange(c) for c in counts]),
                        "delay_ns": raw[:, 2],
                        "gain_re": 10.0 ** (raw[:, 0] / (2 * GAIN_DB_DIVISOR)) * np.cos(raw[:, 1]),
                        "gain_im": 10.0 ** (raw[:, 0] / (2 * GAIN_DB_DIVISOR)) * np.sin(raw[:, 1]),
                        "gain_db": raw[:, 0],
                        "aoa_az_deg": np.degrees(raw[:, 5]),
                        "aoa_el_deg": 90.0 - np.degrees(raw[:, 6]),
                        "aod_az_deg": np.degrees(raw[:, 3]),
                        "aod_el_deg": 90.0 - np.degrees(raw[:, 4]),
                        "interaction_type": "",
                        "interaction_xyz": "",
                        "material_id": "",
                    })[PATH_COLUMNS_OUT]
                    pdf.to_csv(paths_path, mode="w" if first_paths else "a",
                               header=first_paths, index=False, float_format="%.6g")
                    first_paths = False

                n_floors_written += 1

        print(f"  {info.tag:38s} 已写出 {n_floors_written} 个场景")

    pd.DataFrame(meta_rows).to_csv(outdir / "scene_meta.csv", index=False,
                                   float_format="%.4f")
    print(f"\n完成：{links_path}")
    print(f"       {outdir / 'scene_meta.csv'}")
    if with_paths:
        print(f"       {paths_path}")


def normalize_synthetic(raw_dir: Path, outdir: Path) -> None:
    """把自研 RT 引擎导出的原始 CSV 转换为统一字段。

    期望 ``raw_dir`` 下每个场景一个子目录，内含
    ``links_raw.csv``（必需）与 ``paths_raw.csv``（可选）。
    字段映射见 docs/rt_output_format.md。
    """
    if not raw_dir.exists():
        print(f"原始目录不存在: {raw_dir}", file=sys.stderr)
        return
    scenes = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not scenes:
        print(f"{raw_dir} 下没有场景子目录", file=sys.stderr)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    links_path = outdir / "links.csv"
    paths_path = outdir / "paths.csv"
    first_l = first_p = True
    meta_rows: list[dict] = []

    for scene in scenes:
        lp = scene / "links_raw.csv"
        if not lp.exists():
            print(f"  跳过 {scene.name}：缺少 links_raw.csv")
            continue
        raw = pd.read_csv(lp)
        missing = {"rx_x", "rx_y", "rx_z", "tx_x", "tx_y", "tx_z"} - set(raw.columns)
        if missing:
            print(f"  跳过 {scene.name}：缺少必需列 {sorted(missing)}")
            continue

        df = pd.DataFrame({
            "scene_id": raw.get("scene_id", scene.name),
            "dataset": raw.get("dataset", "selfRT"),
            "split": raw.get("split", "custom"),
            "floor_id": 0,
            "floor_idx": raw.get("scene_index", 0),
            "tx_id": raw["tx_id"].astype(int),
            "rx_id": raw["rx_id"].astype(int),
            "rx_x": raw["rx_x"], "rx_y": raw["rx_y"], "rx_z": raw["rx_z"],
            "tx_x": raw["tx_x"], "tx_y": raw["tx_y"], "tx_z": raw["tx_z"],
            "rss_db": raw["rss_db"],
            "tof_ns": raw.get("tof_ns", np.nan),
            "tof_strongest_ns": raw.get("tof_strongest_ns", np.nan),
            "path_count": raw.get("path_count", np.nan),
            "delay_spread_ns": raw.get("delay_spread_ns", np.nan),
            "aoa_az_deg": raw.get("aoa_az_deg", np.nan),
            "aoa_el_deg": raw.get("aoa_el_deg", np.nan),
            "aod_az_deg": raw.get("aod_az_deg", np.nan),
            "aod_el_deg": raw.get("aod_el_deg", np.nan),
            "los_flag": raw.get("los_flag", np.nan),
            "d_geo_m": np.linalg.norm(
                raw[["rx_x", "rx_y", "rx_z"]].to_numpy()
                - raw[["tx_x", "tx_y", "tx_z"]].to_numpy(), axis=1),
        })[LINK_COLUMNS]
        df.to_csv(links_path, mode="w" if first_l else "a",
                  header=first_l, index=False, float_format="%.4f")
        first_l = False

        pp = scene / "paths_raw.csv"
        if pp.exists():
            praw = pd.read_csv(pp)
            pcols = {
                "scene_id": praw.get("scene_id", scene.name),
                "tx_id": praw["tx_id"].astype(int),
                "rx_id": praw["rx_id"].astype(int),
                "path_id": praw["path_id"].astype(int),
                "delay_ns": praw["delay_ns"],
                "gain_re": praw.get("gain_re", np.nan),
                "gain_im": praw.get("gain_im", np.nan),
                "gain_db": praw.get("gain_db", np.nan),
                "aoa_az_deg": praw.get("aoa_az_deg", np.nan),
                "aoa_el_deg": praw.get("aoa_el_deg", np.nan),
                "aod_az_deg": praw.get("aod_az_deg", np.nan),
                "aod_el_deg": praw.get("aod_el_deg", np.nan),
                "interaction_type": praw.get("interaction_type", ""),
                "interaction_xyz": praw.get("interaction_xyz", ""),
                "material_id": praw.get("material_id", ""),
            }
            pd.DataFrame(pcols)[PATH_COLUMNS_OUT].to_csv(
                paths_path, mode="w" if first_p else "a",
                header=first_p, index=False, float_format="%.6g")
            first_p = False

        span = (df[["rx_x", "rx_y"]].max() - df[["rx_x", "rx_y"]].min()).to_numpy()
        meta_rows.append(dict(
            scene_id=str(df["scene_id"].iloc[0]), dataset=str(df["dataset"].iloc[0]),
            split=str(df["split"].iloc[0]), floor_idx=0,
            n_channel=len(df), n_tx=int(df["tx_id"].nunique()),
            n_rx_pos=int(df["rx_id"].nunique()),
            rx_count_per_tx=int(df.groupby("tx_id").size().max()),
            los_ratio=float(df["los_flag"].mean()) if df["los_flag"].notna().any() else np.nan,
            rss_median_db=float(df["rss_db"].median()),
            rss_min_db=float(df["rss_db"].min()), rss_max_db=float(df["rss_db"].max()),
            rss_span_db=float(df["rss_db"].max() - df["rss_db"].min()),
            ds_median_ns=float(df["delay_spread_ns"].median()),
            rx_extent_x_m=float(span[0]), rx_extent_y_m=float(span[1]),
            rx_bbox_area_m2=float(span[0] * span[1]),
        ))
        print(f"  已转换 {scene.name}: {len(df)} 条链路")

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(outdir / "scene_meta.csv", index=False,
                                       float_format="%.4f")
    print(f"\n完成：{links_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["public", "synthetic"], required=True)
    ap.add_argument("--public-root", type=Path, default=Path("data/public_raw"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/synthetic/raw"))
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="如 Wi3R WiPTR")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="如 val_ood val_grid train_grid main_5tx")
    ap.add_argument("--max-floors-per-split", type=int, default=None,
                    help="每个划分最多转换的场景数，用于小样本冒烟测试")
    ap.add_argument("--with-paths", action="store_true", help="同时输出 paths.csv")
    ap.add_argument("--max-path-floors", type=int, default=3,
                    help="逐路径表最多覆盖的场景数，控制文件体积")
    a = ap.parse_args()

    if a.source == "public":
        normalize_public(a.public_root, a.outdir, a.datasets, a.splits,
                         a.max_floors_per_split, a.with_paths, a.max_path_floors)
    else:
        normalize_synthetic(a.raw_dir, a.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
