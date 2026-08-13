import glob
import os
import numpy as np
import pandas as pd

"""Proper permutation test for the overnight-sentiment result.

Three shuffles were not enough, and the shuffle scheme was wrong: permuting
sentiment WITHIN a date preserves that date's average, so any market-wide
component survives it. The null we actually need destroys the TIME ALIGNMENT -
does sentiment attach to the right session? - so sentiment is permuted within
each company, across its own dates.

Company fixed effects are applied by within-transformation (demeaning per
company) rather than dummies, which gives the identical slope coefficient and
makes 500 refits cheap.
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
N_PERM = 500
CTRL = ['prev_return', 'prev_gap', 'prev_intraday', 'log_n']


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
    px['daily_return'] = px['Close'] / px['prev_close'] - 1
    px['prev_return'] = g['daily_return'].shift(1)
    px['prev_gap'] = g['gap'].shift(1)
    px['prev_intraday'] = g['intraday'].shift(1)
    return px


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
d = d[d['n'] >= 3].copy()
d['log_n'] = np.log(d['n'])
d = d.dropna(subset=['gap', 'sent'] + CTRL).reset_index(drop=True)
d['ticker'] = d['ticker'].astype(str)
print(f'n = {len(d):,} company-sessions, {d["ticker"].nunique()} companies\n')


def within(df, cols):
    """Demean each column by company - equivalent to company fixed effects."""
    out = df[cols].copy()
    for c in cols:
        out[c] = df[c] - df.groupby('ticker')[c].transform('mean')
    return out.values


def slope(sent_values):
    tmp = d.copy()
    tmp['sent'] = sent_values
    Y = within(tmp, ['gap'])[:, 0]
    X = within(tmp, ['sent'] + CTRL)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta[0]


real = slope(d['sent'].values)
print(f'Observed coefficient: {real:+.6f}')

rng = np.random.default_rng(11)
null = np.empty(N_PERM)
groups = d.groupby('ticker').indices
for i in range(N_PERM):
    s = d['sent'].values.copy()
    for _, idx in groups.items():          # shuffle each company's own history
        s[idx] = rng.permutation(s[idx])
    null[i] = slope(s)

p_emp = (np.abs(null) >= abs(real)).mean()
print(f'\n=== Permutation null ({N_PERM} shuffles within company, across dates) ===')
print(f'  null mean: {null.mean():+.6f}   sd: {null.std():.6f}')
print(f'  null 95th percentile of |coef|: {np.percentile(np.abs(null), 95):.6f}')
print(f'  observed: {real:+.6f}  ->  {abs(real)/null.std():.1f} null standard deviations')
print(f'  empirical p-value: {p_emp:.4f}   ({int((np.abs(null) >= abs(real)).sum())}'
      f' of {N_PERM} shuffles reached it)')
print('\n  VERDICT:', 'SURVIVES - the timing genuinely matters'
      if p_emp < 0.01 else 'FAILS - could be produced by chance alignment')
