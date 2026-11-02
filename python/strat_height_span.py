"""Reproduce the height and anchor-span stratifications of Section IV-B.

Two claims are covered:
  1. median s_z and median mean|u_z| by receiver height, WiPTR test split
     (values quoted: 0.79 / 0.90 / 0.76 and 0.13 / 0.11 / 0.15)
  2. median s_z, d_eff and kappa for near-coplanar (anchor z span <= 0.3 m)
     against clearly non-coplanar (span > 0.5 m) floors of the three WiPTR
     splits, restricted to floors carrying at least two anchors
     (values quoted: 0.886-0.895 against 0.873-0.874,
      1.42 against 1.55-1.57, 66-76 against 48-49)

Run from rt_localization/python with the project interpreter:
  python strat_height_span.py
"""
import os
import numpy as np
import pandas as pd

RES = os.path.join('..', 'results')
RAW = os.path.join('..', 'data', 'public_raw', 'WiPTR')

# ---- 1. height profile, single continuous split -------------------------
d = pd.read_csv(os.path.join(RES, 'c2full_WiPTR_test.csv'),
                usecols=['rx_z', 'share_z', 'mean_abs_uz'])
b = pd.cut(d['rx_z'], bins=np.arange(0.15, 2.36, 0.1))
g = d.groupby(b, observed=True).agg(n=('share_z', 'size'),
                                    s_z=('share_z', 'median'),
                                    uz=('mean_abs_uz', 'median'))
print('WiPTR test: median s_z and median mean|u_z| by 0.1 m receiver-height bin')
print(g[g['n'] >= 500].to_string())
print('bottom / middle / top:')
for lab, m in [('0.15-0.25', d['rx_z'] <= 0.25),
               ('0.85-1.25', (d['rx_z'] >= 0.85) & (d['rx_z'] <= 1.25)),
               ('2.25-2.35', d['rx_z'] >= 2.25)]:
    s = d[m]
    print('  %-10s n=%6d s_z=%.3f uz=%.3f' % (lab, len(s), s['share_z'].median(),
                                              s['mean_abs_uz'].median()))

# ---- 2. anchor span, floors with at least two anchors -------------------
import h5py  # noqa: E402

for sp in ['test', 'train', 'val']:
    with h5py.File(os.path.join(RAW, sp, 'dataset_sparse.h5'), 'r') as f:
        txz = f['tx'][:, 2]
        fic = f['floor_idx_channels'][:]
        starts = fic[:, 1]
        ends = np.append(starts[1:], f['channels'].shape[0])
        span = np.full(len(fic), np.nan)
        ntx = np.zeros(len(fic), int)
        for fi in range(len(fic)):
            ids = np.unique(f['channels'][starts[fi]:ends[fi], 0])
            z = txz[ids]
            span[fi] = z.max() - z.min()
            ntx[fi] = len(ids)

    s = pd.DataFrame({'floor': np.arange(len(fic)), 'span': span, 'ntx': ntx})
    c = pd.read_csv(os.path.join(RES, 'c2full_WiPTR_%s.csv' % sp),
                    usecols=['floor', 'share_z', 'd_eff', 'kappa'])
    m = c.merge(s, on='floor')
    multi = m['ntx'] >= 2
    print('\n%s: floors with ntx>=2: %d, strictly coplanar among them: %d'
          % (sp, s['ntx'].ge(2).sum(), int(((s['ntx'] >= 2) & (s['span'] < 1e-3)).sum())))
    for lab, mask in [('near-coplanar 0<span<=0.3', multi & (m['span'] > 1e-3) & (m['span'] <= 0.3)),
                      ('non-coplanar span>0.5', multi & (m['span'] > 0.5))]:
        sub = m[mask]
        print('  %-24s n=%7d s_z=%.3f d_eff=%.2f kappa=%.1f'
              % (lab, len(sub), sub['share_z'].median(), sub['d_eff'].median(),
                 sub['kappa'].median()))
