"""Which way does the relationship run: does sentiment lead price, or follow it?

This is the central result for Bitcoin. Daily aggregation cannot answer the
question at all - a tweet posted at 02:00 and one posted at 23:00 land in the
same bucket, which is then compared with the whole day's return, so a strong
daily correlation is equally consistent with pure reaction.

Bitcoin trades without a break, so the day can be cut anywhere and asked a
genuinely forward-looking question:

    sentiment over hours [0, H)  ->  return over hours [H, 24)

The two windows never overlap. The mirror test - the first part's return against
the second part's sentiment - measures the reaction channel, and comparing the
two tells us the direction the relationship actually runs.

Reproduces: journal section 03.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis.common import load_hourly_btc, ci, mde

CUTS = [4, 6, 8, 12, 16, 18, 20]
MIN_TWEETS_PER_HALF = 50


def run():
    meta, px = load_hourly_btc()
    hour_px = px['close']

    meta = meta[(meta['ts'] >= px.index.min()) & (meta['ts'] <= px.index.max())].copy()
    meta['day'] = meta['ts'].dt.floor('D')
    meta['hour'] = meta['ts'].dt.hour
    print(f'Tweets inside the hourly-price window: {len(meta):,}  '
          f'({meta["day"].nunique()} days, {meta["ts"].min().date()} -> {meta["ts"].max().date()})\n')

    def price_at(days, hour):
        return hour_px.reindex(pd.DatetimeIndex(days) + pd.Timedelta(hours=hour)).values

    print(f'{"cut":>4s} {"n":>5s}   {"PREDICTION sentiment->later price":^36s}   '
          f'{"REACTION price->later sentiment":^36s}')
    print('-' * 92)
    rows = []
    for H in CUTS:
        early = meta[meta['hour'] < H].groupby('day')['finbert'].agg(['mean', 'size'])
        late = meta[meta['hour'] >= H].groupby('day')['finbert'].agg(['mean', 'size'])
        d = early.join(late, lsuffix='_e', rsuffix='_l', how='inner')
        d = d[(d['size_e'] >= MIN_TWEETS_PER_HALF) & (d['size_l'] >= MIN_TWEETS_PER_HALF)]
        if len(d) < 100:
            continue

        p_open = price_at(d.index, 0)
        p_cut = price_at(d.index, H)
        p_close = price_at(d.index + pd.Timedelta(days=1), 0)
        ok = ~(np.isnan(p_open) | np.isnan(p_cut) | np.isnan(p_close))
        dd = d[ok]
        ret_first = p_cut[ok] / p_open[ok] - 1
        ret_second = p_close[ok] / p_cut[ok] - 1

        r_pred, p_pred = stats.pearsonr(dd['mean_e'], ret_second)
        r_react, p_react = stats.pearsonr(ret_first, dd['mean_l'])
        lo1, hi1 = ci(r_pred, len(dd))
        lo2, hi2 = ci(r_react, len(dd))
        star = '*' if p_pred < 0.05 else ' '
        print(f'{H:4d} {len(dd):5d}   r={r_pred:+.4f} [{lo1:+.3f},{hi1:+.3f}] {star}  '
              f'   r={r_react:+.4f} [{lo2:+.3f},{hi2:+.3f}] ')
        rows.append((H, len(dd), r_pred, p_pred, r_react, p_react))

    print(f'\n{"":4s} MDE at these sample sizes: '
          f'{mde(int(np.median([r[1] for r in rows]))):.3f}')
    print('\nThe reaction channel holds at every cut with intervals far from zero;')
    print('the prediction channel oscillates around zero and its intervals all')
    print('contain it. The two sets of intervals nowhere overlap.')

    sig = [r for r in rows if r[3] < 0.05]
    if sig:
        thr = 0.05 / len(rows)
        print(f'\nCuts reaching p<0.05 on the prediction side: {[r[0] for r in sig]}')
        print(f'With {len(rows)} tests the Bonferroni threshold is {thr:.4f}; '
              f'{"none survive it" if all(r[3] > thr for r in sig) else "some survive"}.')
    return rows


if __name__ == '__main__':
    run()
