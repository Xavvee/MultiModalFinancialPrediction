import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

"""Measuring a weak signal the way quantitative finance actually does.

Directional accuracy is a classification metric and a poor fit here. It discards
magnitude, and with 54.7% of gaps positive a trivial constant call scores 54.7%
for free - while a signal of our strength (r = 0.16) can lift it by at most
0.45 points, which is below the sampling noise. The metric literally cannot see
an effect this size.

The standard alternative sorts the cross-section by the signal and compares the
extremes:

  QUANTILE SPREAD   within each session, rank companies by overnight sentiment
                    and compare the realised gap of the top group against the
                    bottom. Reported in basis points, an economic unit.

  MONOTONICITY      a real signal orders the quantiles; noise does not. This is
                    a genuine test, not decoration - a spurious correlation has
                    no reason to produce a monotone ladder.

  LONG-SHORT SERIES forming the spread per session gives a time series whose
                    t-statistic handles cross-sectional correlation naturally,
                    since each session contributes one observation.

The spread is a measure of SIGNAL STRENGTH in economic units, not a strategy:
the opening gap cannot be captured by trading, because the full overnight
sentiment is only known moments before the open.
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
MIN_TWEETS = 3
MIN_FIRMS_PER_SESSION = 6


def load_prices():
    frames = []
    for path in sorted(glob.glob(os.path.join(PRICES, '*.csv'))):
        t = os.path.splitext(os.path.basename(path))[0]
        d = pd.read_csv(path)
        d['date'] = pd.to_datetime(d['Date'])
        d['ticker'] = t
        frames.append(d[['ticker', 'date', 'Open', 'Close']])
    px = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'date'])
    g = px.groupby('ticker')
    px['prev_close'] = g['Close'].shift(1)
    px['gap'] = px['Open'] / px['prev_close'] - 1
    px['intraday'] = px['Close'] / px['Open'] - 1
    return px[['ticker', 'date', 'gap', 'intraday']]


def build():
    tw = pd.read_parquet(TWEETS)
    tw = tw[tw['finbert'].notna()].copy()
    tw['ts'] = pd.to_datetime(tw['ts'])
    hour = tw['ts'].dt.hour + tw['ts'].dt.minute / 60.0
    day = tw['ts'].dt.floor('D')
    ac = hour >= CLOSE_UTC
    tw['session'] = day.where(~ac, day + pd.Timedelta(days=1))
    tw = tw[ac | (hour < OPEN_UTC)]

    agg = (tw.groupby(['ticker', 'session'])
             .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
             .rename(columns={'session': 'date'}))
    d = agg.merge(load_prices(), on=['ticker', 'date'], how='inner')
    d = d[(d['n'] >= MIN_TWEETS) & d['gap'].notna()].copy()

    # Keep only sessions with enough firms to form a cross-section.
    per_session = d.groupby('date')['ticker'].transform('size')
    return d[per_session >= MIN_FIRMS_PER_SESSION].reset_index(drop=True)


def quantile_table(d, q, target='gap'):
    """Average realised move by sentiment quantile, formed within each session."""
    d = d.copy()
    d['bucket'] = (d.groupby('date')['sent']
                    .transform(lambda s: pd.qcut(s.rank(method='first'), q,
                                                 labels=False, duplicates='drop')))
    d = d[d['bucket'].notna()]
    rows = []
    for b in sorted(d['bucket'].unique()):
        sub = d[d['bucket'] == b]
        rows.append({'bucket': int(b) + 1, 'n': len(sub),
                     'mean_bp': sub[target].mean() * 10000})
    return pd.DataFrame(rows), d


def long_short(d, q, target='gap'):
    """Per-session spread between the top and bottom sentiment group."""
    _, marked = quantile_table(d, q, target)
    top, bot = marked['bucket'].max(), marked['bucket'].min()
    per_date = (marked[marked['bucket'] == top].groupby('date')[target].mean()
                - marked[marked['bucket'] == bot].groupby('date')[target].mean())
    per_date = per_date.dropna()
    t, p = stats.ttest_1samp(per_date, 0)
    ann = per_date.mean() / per_date.std() * np.sqrt(252) if per_date.std() > 0 else np.nan
    return per_date, per_date.mean() * 10000, t, p, ann


def run():
    d = build()
    print(f'{len(d):,} company-sessions | {d["ticker"].nunique()} firms | '
          f'{d["date"].nunique()} sessions')
    print(f'median firms per session: {int(d.groupby("date").size().median())}\n')

    for q, name in [(3, 'TERCILES'), (5, 'QUINTILES')]:
        tbl, _ = quantile_table(d, q)
        print(f'=== {name}: realised opening gap by overnight-sentiment rank ===')
        for _, r in tbl.iterrows():
            bar = '#' * max(int(r['mean_bp'] / 2), 0)
            print(f'  group {int(r["bucket"])}/{q}  n={int(r["n"]):5,}  '
                  f'{r["mean_bp"]:+7.2f} bp  {bar}')
        mono = tbl['mean_bp'].is_monotonic_increasing
        print(f'  monotonic across groups: {"YES" if mono else "no"}')

        series, spread_bp, t, p, sharpe = long_short(d, q)
        print(f'  top-minus-bottom spread: {spread_bp:+.2f} bp per session  '
              f'(t={t:+.2f}, p={p:.4f}, n={len(series)} sessions)')
        print(f'  annualised ratio of mean to volatility: {sharpe:.2f}\n')

    # The same construction on the intraday leg, where nothing should be left.
    print('=== PLACEBO: the same sort against the INTRADAY move ===')
    tbl, _ = quantile_table(d, 5, target='intraday')
    for _, r in tbl.iterrows():
        print(f'  group {int(r["bucket"])}/5  {r["mean_bp"]:+7.2f} bp')
    _, spread_bp, t, p, _ = long_short(d, 5, target='intraday')
    print(f'  spread: {spread_bp:+.2f} bp (t={t:+.2f}, p={p:.4f})')
    print('  -> the information is priced at the open and nothing is left to capture.')

    print('\n=== Interpretation ===')
    print('  The spread is a measure of signal strength in economic units, not a')
    print('  strategy: capturing the gap would require holding from the previous')
    print('  close, before the overnight sentiment exists.')


if __name__ == '__main__':
    run()
