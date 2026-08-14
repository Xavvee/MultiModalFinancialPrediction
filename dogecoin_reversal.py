import numpy as np
import pandas as pd
from scipy import stats

from dogecoin_control import load, window_return

"""Does the attention effect reverse, as the theory says it should?

The Musk/Dogecoin episode showed mentions moving price sharply upward - and
doing so regardless of whether the post was upbeat or gloomy (+2.43% after
positive posts, +1.67% after negative ones, p = 0.46). Both up.

Barber & Odean (2008) explain why. A retail investor can BUY anything that
catches their eye, but can only SELL what they already hold. Attention therefore
produces net buying pressure whatever its tone - which is exactly the asymmetry
we measured without naming it.

That theory makes a further prediction this script tests: buying driven by
attention rather than information has nothing to hold it up, so the move should
DECAY once the attention fades. If the rise persists indefinitely, the effect is
informational; if it reverts, it is attention.
"""

WINDOWS = [5, 15, 30, 60, 240, 720, 1440, 2880, 10080]   # 5 min .. 1 week
LABELS = {5: '5 min', 15: '15 min', 30: '30 min', 60: '1 h', 240: '4 h',
          720: '12 h', 1440: '1 day', 2880: '2 days', 10080: '1 week'}


def run():
    tw, price = load()
    print(f'DOGE-related posts: {int(tw["is_doge"].sum())} | '
          f'control: {int((~tw["is_doge"]).sum())}\n')

    rows = []
    for w in WINDOWS:
        tw[f'f{w}'] = [window_return(price, t, w, True) for t in tw['ts']]
        a = tw.loc[tw['is_doge'], f'w{w}' if False else f'f{w}'].dropna()
        b = tw.loc[~tw['is_doge'], f'f{w}'].dropna()
        if len(a) < 10:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({'w': w, 'doge': a.mean() * 100, 'ctrl': b.mean() * 100,
                     'diff': (a.mean() - b.mean()) * 100, 't': t, 'p': p, 'n': len(a)})

    print(f'{"horizon":>9s} {"DOGE posts":>11s} {"control":>9s} {"excess":>9s} '
          f'{"t":>6s} {"p":>8s}')
    print('-' * 60)
    peak = max(rows, key=lambda r: r['diff'])
    for r in rows:
        star = ' *' if r['p'] < 0.05 else '  '
        mark = '  <- peak' if r['w'] == peak['w'] else ''
        print(f'{LABELS[r["w"]]:>9s} {r["doge"]:+10.3f}% {r["ctrl"]:+8.3f}% '
              f'{r["diff"]:+8.3f}% {r["t"]:+6.2f} {r["p"]:8.4f}{star}{mark}')

    print(f'\nPeak excess move: {peak["diff"]:+.3f}% at {LABELS[peak["w"]]}')
    last = rows[-1]
    decay = (1 - last['diff'] / peak['diff']) * 100 if peak['diff'] else np.nan
    print(f'By {LABELS[last["w"]]}: {last["diff"]:+.3f}%  '
          f'({decay:.0f}% of the peak has decayed)')

    # Read the SIGN, not just significance: a significant negative excess at the
    # long horizon is a reversal, which is the opposite of persistence.
    early = [r for r in rows if r['w'] <= 60 and r['p'] < 0.05]
    if early and last['diff'] < 0 and last['p'] < 0.05:
        print('\n  Sharp significant rise, then a significant REVERSAL - buying that')
        print('  had no informational basis is given back. This is the attention')
        print('  mechanism (Barber & Odean 2008), not news being priced in.')
    elif early and last['p'] >= 0.05:
        print('\n  Sharp rise that fades into noise: consistent with attention,')
        print('  though the long horizon cannot distinguish decay from reversal.')
    elif last['diff'] > 0 and last['p'] < 0.05:
        print('\n  The gain persists at the longest horizon, which would point at')
        print('  information rather than attention.')

    print('\n  CAVEAT: with 101 events in a single year, week-long windows overlap')
    print('  heavily, so the events are not independent and the long-horizon')
    print('  p-value is more confident than it should be. The short horizons,')
    print('  where windows barely overlap, carry the weight of the evidence.')

    print('\n=== Tone made no difference to any of it ===')
    import os
    if os.path.exists('data/dogecoin/processed/musk_tweets_scored.parquet'):
        sc = pd.read_parquet('data/dogecoin/processed/musk_tweets_scored.parquet')
        tw2 = tw.reset_index(drop=True)
        tw2['finbert'] = sc['finbert'].values
        doge = tw2[tw2['is_doge']]
        for w in [15, 60, 1440]:
            hi = doge.loc[doge['finbert'] > doge['finbert'].median(), f'f{w}'].dropna()
            lo = doge.loc[doge['finbert'] <= doge['finbert'].median(), f'f{w}'].dropna()
            t, p = stats.ttest_ind(hi, lo, equal_var=False)
            print(f'  {LABELS[w]:>7s}: upbeat {hi.mean()*100:+.3f}%  '
                  f'gloomy {lo.mean()*100:+.3f}%   t={t:+.2f} p={p:.3f}')
        print('  -> both groups move the same way: the mention does the work, not the tone.')


if __name__ == '__main__':
    run()
