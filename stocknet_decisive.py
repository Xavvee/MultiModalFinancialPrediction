import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

"""Is the overnight-sentiment result real, or the previous day's move echoing?

Overnight sentiment correlates with the opening gap (t=+6.47). It ALSO correlates
with the previous session's return (r=+0.092) - people tweet about what just
happened. If the previous return itself moves the next open, sentiment would
track the gap while carrying nothing of its own.

The decisive test therefore controls for what was already known before the
tweets were written. Two further checks:

  SPLIT TARGET  the gap is priced at the open; the intraday move that follows is
                tradeable all session. Genuine overnight information should show
                up in the GAP and NOT in the intraday leg - anything predicting
                both equally is more likely a stale-price artifact.
  PLACEBO       sentiment must not "predict" the gap that already happened.
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
MIN_TWEETS = 3


def load_prices():
    frames = []
    for path in sorted(glob.glob(os.path.join(PRICES, '*.csv'))):
        t = os.path.splitext(os.path.basename(path))[0]
        d = pd.read_csv(path)
        d['date'] = pd.to_datetime(d['Date'])
        d['ticker'] = t
        frames.append(d[['ticker', 'date', 'Open', 'Close', 'Volume']])
    px = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'date'])
    g = px.groupby('ticker')
    px['prev_close'] = g['Close'].shift(1)
    px['gap'] = px['Open'] / px['prev_close'] - 1
    px['intraday'] = px['Close'] / px['Open'] - 1
    px['daily_return'] = px['Close'] / px['prev_close'] - 1
    px['prev_return'] = g['daily_return'].shift(1)
    px['prev_gap'] = g['gap'].shift(1)
    px['prev_intraday'] = g['intraday'].shift(1)
    px['prev_return2'] = g['daily_return'].shift(2)
    return px


tw = pd.read_parquet(TWEETS)
tw = tw[tw['finbert'].notna()].copy()
tw['ts'] = pd.to_datetime(tw['ts'])
hour = tw['ts'].dt.hour + tw['ts'].dt.minute / 60.0
day = tw['ts'].dt.floor('D')
after_close = hour >= CLOSE_UTC
tw['session'] = day.where(~after_close, day + pd.Timedelta(days=1))
tw['closed'] = after_close | (hour < OPEN_UTC)

night = tw[tw['closed']]
agg = (night.groupby(['ticker', 'session'])
            .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
            .rename(columns={'session': 'date'}))

d = agg.merge(load_prices(), on=['ticker', 'date'], how='inner')
d = d[d['n'] >= MIN_TWEETS].copy()
d['sent_z'] = (d['sent'] - d['sent'].mean()) / d['sent'].std()
d['log_n'] = np.log(d['n'])
print(f'company-sessions with >={MIN_TWEETS} overnight tweets: {len(d):,}\n')

fe = pd.get_dummies(d['ticker'], prefix='t', drop_first=True).astype(float)
clusters = d['date'].factorize()[0]


def run(label, target, controls):
    X = fe.copy()
    X['sent_z'] = d['sent_z'].values
    X['log_n'] = d['log_n'].values
    for c in controls:
        X[c] = d[c].values
    X = sm.add_constant(X)
    y = d[target]
    ok = X.notna().all(axis=1).values & y.notna().values
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='cluster',
                                 cov_kwds={'groups': clusters[ok]})
    c_, se, p = m.params['sent_z'], m.bse['sent_z'], m.pvalues['sent_z']
    tag = 'SIGNIFICANT' if p < 0.05 else 'dead'
    print(f'  {label:52s} coef={c_:+.6f} t={c_/se:+6.2f} p={p:.4f}  {tag}  n={int(m.nobs):,}')
    return p


print('=== Does the gap result survive knowing what already happened? ===')
run('gap ~ sentiment (no controls)', 'gap', [])
run('gap ~ sentiment + previous return', 'gap', ['prev_return'])
run('gap ~ sentiment + prev return, gap, intraday', 'gap',
    ['prev_return', 'prev_gap', 'prev_intraday'])
p_full = run('gap ~ sentiment + all of the above + return t-2', 'gap',
             ['prev_return', 'prev_gap', 'prev_intraday', 'prev_return2'])

print('\n=== SPLIT TARGET: where does the information land? ===')
p_gap = run('-> opening GAP        (priced while shut)', 'gap',
            ['prev_return', 'prev_gap', 'prev_intraday'])
p_intra = run('-> INTRADAY move     (tradeable all session)', 'intraday',
              ['prev_return', 'prev_gap', 'prev_intraday'])

print('\n=== PLACEBO: sentiment vs the gap that already happened ===')
p_pla = run('-> PREVIOUS gap', 'prev_gap', ['prev_return', 'prev_return2'])

print('\n=== READING ===')
if p_full < 0.05 and p_pla >= 0.05:
    print('  Survives full controls, placebo dead -> genuine overnight information.')
    if p_gap < 0.05 and p_intra >= 0.05:
        print('  And it lands in the gap but not the intraday leg - exactly the')
        print('  pattern expected if the market prices it at the open.')
elif p_full >= 0.05:
    print('  Dies once the previous session is controlled for -> the apparent')
    print('  prediction was the previous move echoing through sentiment.')
else:
    print('  Placebo alive -> sentiment tracks price symmetrically; not predictive.')
