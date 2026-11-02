"""Recompute the scan-point correlations and bin ratios quoted in Section 4.9.

For each noise setting the end-to-end run stored one row per Rx position in
results/e2e_sens_<tag>/e2e_links.csv. The per-position total vertical error is the
quadrature sum of the signed error and the Monte Carlo spread, which is the
convention used by end_to_end_compare.py (columns aoa_tz / wls_tz). The scan
directories predate those columns, so it is rebuilt here from aoa_ez and aoa_sz.

The paper reports these statistics over all positions, not over the
criterion-passing subset; both are printed below so the two readings cannot be
confused with each other.

Paper values (Sec. 4.9), over all positions:
  rank correlation with the total vertical error
      vertical share        0.33 to 0.34
      mean vertical component  -0.38 to -0.40
      condition number      0.35 to 0.36
  ratio of the median vertical error between the s_z>0.5 and the s_z<=0.35 bin
      1.54 to 1.62 under every setting
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
TAGS = ['sd005', 'sd020', 'th025', 'th10']
QUANT = [('share_z', 'vertical share'),
         ('mean_abs_uz', 'mean |u_z|'),
         ('kappa', 'condition number'),
         ('d_eff', 'effective dimension')]


def load(path):
    d = pd.read_csv(path)
    if 'aoa_tz' not in d.columns:
        d['aoa_tz'] = np.hypot(d['aoa_ez'], d['aoa_sz'])
    return d


def report(label, d):
    y = d['aoa_tz'].to_numpy()
    print('  --- %s  n=%d, median %.4f' % (label, len(d), np.median(y)))
    for col, name in QUANT:
        print('    %-18s spearman %+0.3f' % (name, spearmanr(d[col], y).correlation))
    low = np.median(y[d['share_z'].to_numpy() <= 0.35])
    high = np.median(y[d['share_z'].to_numpy() > 0.5])
    print('    bin ratio %.3f  (%.4f / %.4f)' % (high / low, high, low))


def main():
    for tag in TAGS:
        path = os.path.join(RES, 'e2e_sens_%s' % tag, 'e2e_links.csv')
        if not os.path.exists(path):
            print('%s missing' % path)
            continue
        d = load(path)
        print('=== %s ===' % tag)
        report('all positions', d)
        report('criterion pass', d[(d['rank'] == 3) & (d['kappa'] < 100)])


if __name__ == '__main__':
    main()
