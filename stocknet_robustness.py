import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

"""Full scrutiny of the overnight-sentiment result before it is believed.

The finding: sentiment posted while the market is shut predicts the opening gap
(t=+6.2 with company fixed effects and date-clustered errors), survives controls
for everything already known, and lands in the gap but not the intraday leg.

Everything here is an attempt to break it:
  - the other sentiment model (FinBERT is finance-trained, RoBERTa is not)
  - every density threshold, since sparse company-days are noisy
  - winsorised returns, so a handful of earnings jumps cannot carry it
  - each half of the sample separately, as a crude out-of-sample check
  - excluding the largest / most-tweeted companies, in case one name drives it
  - a randomisation check: shuffle sentiment within each date and it must die
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
CTRL = ['prev_return', 'prev_gap', 'prev_intraday']


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
         .agg(finbert=('finbert', 'mean'), roberta=('roberta', 'mean'),
              n=('finbert', 'size')).reset_index()
         .rename(columns={'session': 'date'}))
FULL = agg.merge(load_prices(), on=['ticker', 'date'], how='inner')


def fit(d, key='finbert', target='gap', controls=CTRL, label='', seed=None):
    d = d.copy()
    if len(d) < 200:
        print(f'  {label:50s} (n={len(d)} too small)')
        return
    v = d[key].values
    if seed is not None:                       # randomisation: shuffle within date
        rng = np.random.default_rng(seed)
        d['_s'] = (d.groupby('date')[key]
                     .transform(lambda s: rng.permutation(s.values)))
        v = d['_s'].values
    d['sent_z'] = (v - np.nanmean(v)) / np.nanstd(v)
    X = pd.get_dummies(d['ticker'], prefix='t', drop_first=True).astype(float)
    X['sent_z'] = d['sent_z'].values
    X['log_n'] = np.log(d['n'].values)
    for c in controls:
        X[c] = d[c].values
    X = sm.add_constant(X)
    y = d[target]
    ok = X.notna().all(axis=1).values & y.notna().values
    cl = d['date'].factorize()[0]
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='cluster', cov_kwds={'groups': cl[ok]})
    c_, se, p = m.params['sent_z'], m.bse['sent_z'], m.pvalues['sent_z']
    tag = 'OK  ' if p < 0.05 else 'DEAD'
    print(f'  {label:50s} coef={c_:+.6f} t={c_/se:+6.2f} p={p:.4f} {tag} n={int(m.nobs):,}')


base = FULL[FULL['n'] >= 3]
print(f'Base sample: {len(base):,} company-sessions\n')

print('=== 1. The other sentiment model ===')
fit(base, 'finbert', label='FinBERT (finance-trained)')
fit(base, 'roberta', label='RoBERTa (social-media-trained)')

print('\n=== 2. Density thresholds ===')
for thr in [1, 2, 3, 5, 10, 20]:
    fit(FULL[FULL['n'] >= thr], label=f'>= {thr} overnight tweets')

print('\n=== 3. Winsorised gap (1%/99%) - no single jump carries it ===')
w = base.copy()
lo, hi = w['gap'].quantile([0.01, 0.99])
w['gap'] = w['gap'].clip(lo, hi)
fit(w, label='gap winsorised')

print('\n=== 4. Split sample (crude out-of-sample) ===')
mid = base['date'].quantile(0.5)
fit(base[base['date'] <= mid], label=f'first half (to {pd.Timestamp(mid).date()})')
fit(base[base['date'] > mid], label='second half')

print('\n=== 5. Drop the most-tweeted companies ===')
top = base.groupby('ticker', observed=True)['n'].sum().nlargest(5).index.tolist()
print(f'  (dropping {", ".join(top)})')
fit(base[~base['ticker'].isin(top)], label='excluding top-5 by tweet volume')

print('\n=== 6. Randomisation: shuffle sentiment within each date ===')
for s in [1, 2, 3]:
    fit(base, label=f'shuffled sentiment, seed {s}', seed=s)
print('  (these must be dead - if not, the specification itself manufactures significance)')
