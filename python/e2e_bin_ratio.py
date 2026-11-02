"""Recompute the end-to-end vertical-error bins quoted in Sections 4.7 and 4.9.

Reads results/e2e_links.csv (one row per Rx position, all estimators) and prints
the median vertical error of the accepted positions in two vertical-share bins,
for the AoA and WLS estimators, plus the ratio between them.

Paper values (Sec. 4.7): within the criterion-passing subset, s_z<=0.35 gives
0.276 m and s_z>0.5 gives 0.404 m for AoA (a ratio of 1.46); WLS gives 0.148 m
and 0.325 m (a ratio of 2.19).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, '..', 'results', 'e2e_links.csv')

COLS = None  # unused


def main():
    d = pd.read_csv(E2E, usecols=['rank', 'kappa', 'share_z', 'aoa_tz', 'wls_tz'])
    passed = (d['rank'] == 3) & (d['kappa'] < 100)
    low = passed & (d['share_z'] <= 0.35)
    high = passed & (d['share_z'] > 0.5)
    print('positions total            %7d' % len(d))
    print('criterion pass             %7d' % passed.sum())
    print('pass & s_z<=0.35           %7d' % low.sum())
    print('pass & s_z>0.5             %7d' % high.sum())
    # the stored columns are absolute vertical errors, the *_tz columns
    for est, col in (('AoA', 'aoa_tz'), ('WLS', 'wls_tz')):
        e = d[col].abs()
        a = e[low].median()
        b = e[high].median()
        print('%s  low=%.4f  high=%.4f  ratio=%.3f  diff=%.1f%%'
              % (est, a, b, b / a, 100.0 * (b - a) / b))
    # the full-sample medians quoted at the end of Sec. 4.9
    for est, col in (('AoA', 'aoa_tz'), ('WLS', 'wls_tz')):
        print('%s full-sample median      %.4f' % (est, d[col].abs().median()))


if __name__ == '__main__':
    main()
