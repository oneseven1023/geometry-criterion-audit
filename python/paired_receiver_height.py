"""Paired receiver-height contrast on the val_grid split.

Section IV-B reports a controlled comparison in which the only quantity that
changes is the receiver height, the anchors being fixed at 2.30 m on every
floor.  The val_grid split is the one that allows it: both anchors of each
floor stand at 2.30 m and the receivers occupy exactly two heights, 0.20 m and
2.30 m, on all ten floors.

    python python/paired_receiver_height.py

Input : results/c2full_WiPTR_val_grid.csv  (per-link metrics, c2_fullscale_vec)
Output: results/paired_height.csv and a printed table

The paired difference is the difference of the two medians within one floor,
so that floor-to-floor variation of the scene does not enter it.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
SRC = os.path.join(RESULTS, "c2full_WiPTR_val_grid.csv")
OUT = os.path.join(RESULTS, "paired_height.csv")
OUT_DETAIL = os.path.join(RESULTS, "paired_height_floors.csv")

LOW, HIGH = 0.20, 2.30
TOL = 0.05


def read_rows(path):
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            yield row


def height_bin(rx_z):
    """Snap a receiver height to the nearer of the two occupied levels."""
    if abs(rx_z - LOW) <= TOL:
        return LOW
    if abs(rx_z - HIGH) <= TOL:
        return HIGH
    return None


def group_key(row):
    """One group is one receiver height on one floor; anchors are common."""
    z = height_bin(float(row["rx_z"]))
    if z is None:
        return None
    return (int(row["floor"]), z)


def vdop_over_best_axis(row):
    """Standard deviation of the vertical axis over that of the best axis."""
    var_z = float(row["var_z"])
    lam_max = float(row["lam_max"])
    if lam_max <= 0 or var_z <= 0:
        return float("nan")
    return float(np.sqrt(var_z * lam_max))


def main():
    metrics = ["share_z", "kappa", "d_eff", "mean_abs_uz", "ratio"]
    per_floor = defaultdict(lambda: defaultdict(list))
    for row in read_rows(SRC):
        key = group_key(row)
        if key is None:
            continue
        floor, z = key
        per_floor[(floor, z)]["share_z"].append(float(row["share_z"]))
        per_floor[(floor, z)]["kappa"].append(float(row["kappa"]))
        per_floor[(floor, z)]["d_eff"].append(float(row["d_eff"]))
        per_floor[(floor, z)]["mean_abs_uz"].append(float(row["mean_abs_uz"]))
        per_floor[(floor, z)]["ratio"].append(vdop_over_best_axis(row))

    floors = sorted({f for f, _ in per_floor})
    pooled = {z: {m: [] for m in metrics} for z in (LOW, HIGH)}
    for (floor, z), cols in per_floor.items():
        for m in metrics:
            pooled[z][m].extend(v for v in cols[m] if np.isfinite(v))

    print(f"{'quantity':>14} {LOW:>10} {HIGH:>10} {'paired':>9} {'sign':>6}")
    rows = []
    for m in metrics:
        lo = float(np.median(pooled[LOW][m]))
        hi = float(np.median(pooled[HIGH][m]))
        diffs = []
        for floor in floors:
            a = per_floor.get((floor, LOW), {}).get(m, [])
            b = per_floor.get((floor, HIGH), {}).get(m, [])
            if a and b:
                diffs.append(float(np.median(b)) - float(np.median(a)))
        paired = float(np.median(diffs)) if diffs else float("nan")
        same = sum(1 for d in diffs if np.sign(d) == np.sign(paired))
        print(f"{m:>14} {lo:10.4f} {hi:10.4f} {paired:+9.4f} {same:>3d}/{len(diffs)}")
        rows.append(
            {
                "quantity": m,
                "median_low": lo,
                "median_high": hi,
                "paired_diff": paired,
                "floors_same_sign": f"{same}/{len(diffs)}",
            }
        )

    # per-floor medians, for the paired figure
    detail = []
    for floor in floors:
        for z in (LOW, HIGH):
            cols = per_floor.get((floor, z), {})
            if not cols:
                continue
            rec = {"floor": floor, "rx_z": z, "n": len(cols["share_z"])}
            for m in metrics:
                vals = np.asarray([v for v in cols[m] if np.isfinite(v)])
                rec[m] = float(np.median(vals)) if vals.size else float("nan")
            detail.append(rec)

    with open(OUT_DETAIL, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
