import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

"""Two objections an examiner will raise about the overnight-gap result.

  EARNINGS   Quarterly results are published after the close and produce the
             largest gaps of the year. If commentary swells around them, the
             whole effect could be an earnings proxy rather than a sentiment
             signal. Tested by removing earnings season, and separately by
             removing the largest gaps outright.

  PRE-MARKET US equities trade before the official open. If the effect comes only
             from posts written shortly BEFORE the open, it may be reflecting
             pre-market price moves that daily OHLC data cannot see - which would
             make it reaction wearing prediction's clothes. Tested by splitting
             the closed window into an early part (just after the close, when no
             pre-market trading has happened) and a late part.

The early-window test is the more important of the two: it isolates a period in
which there is genuinely nothing for the sentiment to be reacting to.
"""

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'
CLOSE_UTC, OPEN_UTC = 21, 13.5
CONTROLS = ['prev_return', 'prev_gap', 'prev_intraday', 'log_n']
MIN_TWEETS = 3

# US reporting clusters in the weeks following each quarter end.
EARNINGS_WEEKS = [(1, 10, 2, 20), (4, 10, 5, 20), (7, 10, 8, 20), (10, 10, 11, 20)]


def in_earnings_season(dates):
    out = pd.Series(False, index=dates.index if hasattr(dates, 'index') else None)
    md = pd.DatetimeIndex(dates)
    flag = np.zeros(len(md), dtype=bool)
    for m1, d1, m2, d2 in EARNINGS_WEEKS:
        start = (md.month > m1) | ((md.month == m1) & (md.day >= d1))
        end = (md.month < m2) | ((md.month == m2) & (md.day <= d2))
        flag |= (start & end)
    return flag


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


def tweets_with_window():
    tw = pd.read_parquet(TWEETS)
    tw = tw[tw['finbert'].notna()].copy()
    tw['ts'] = pd.to_datetime(tw['ts'])
    hour = tw['ts'].dt.hour + tw['ts'].dt.minute / 60.0
    day = tw['ts'].dt.floor('D')
    ac = hour >= CLOSE_UTC
    tw['session'] = day.where(~ac, day + pd.Timedelta(days=1))
    tw = tw[ac | (hour < OPEN_UTC)].copy()
    tw['hour'] = hour[tw.index]
    # EARLY: the hours right after the close, before any pre-market trading.
    # LATE: the run-up to the open, when pre-market is already active.
    tw['window'] = np.where(tw['hour'] >= CLOSE_UTC, 'early',
                            np.where(tw['hour'] < 8, 'early', 'late'))
    return tw


def panel(agg, px):
    d = agg.merge(px, on=['ticker', 'date'], how='inner')
    d = d[d['n'] >= MIN_TWEETS].copy()
    d['log_n'] = np.log(d['n'])
    d = d.dropna(subset=['gap', 'sent'] + CONTROLS).copy()
    d['sent_z'] = (d['sent'] - d['sent'].mean()) / d['sent'].std()
    return d


def estimate(d, label, target='gap'):
    if len(d) < 200:
        print(f'  {label:44s} (only {len(d)} rows - skipped)')
        return
    X = pd.get_dummies(d['ticker'], prefix='t', drop_first=True).astype(float)
    X['sent_z'] = d['sent_z'].values
    for c in CONTROLS:
        X[c] = d[c].values
    X = sm.add_constant(X)
    y = d[target]
    ok = X.notna().all(axis=1).values & y.notna().values
    m = sm.OLS(y[ok], X[ok]).fit(cov_type='cluster',
                                 cov_kwds={'groups': d['date'].factorize()[0][ok]})
    c, se, p = m.params['sent_z'], m.bse['sent_z'], m.pvalues['sent_z']
    tag = 'HOLDS' if p < 0.05 else 'gone'
    print(f'  {label:44s} coef={c:+.6f} t={c/se:+5.2f} p={p:.4f} {tag}  n={int(m.nobs):,}')


def main():
    px = load_prices()
    tw = tweets_with_window()
    agg_all = (tw.groupby(['ticker', 'session'])
                 .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
                 .rename(columns={'session': 'date'}))
    d = panel(agg_all, px)
    print(f'Baseline sample: {len(d):,} company-sessions\n')

    print('=== OBJECTION 1: is this just earnings season? ===')
    estimate(d, 'full sample')
    season = in_earnings_season(d['date'])
    print(f'  ({season.mean()*100:.0f}% of rows fall in reporting weeks)')
    estimate(d[~season], 'OUTSIDE earnings season')
    estimate(d[season], 'inside earnings season')

    big = d['gap'].abs() >= d['gap'].abs().quantile(0.95)
    estimate(d[~big], 'excluding the largest 5% of gaps')
    big1 = d['gap'].abs() >= d['gap'].abs().quantile(0.90)
    estimate(d[~big1], 'excluding the largest 10% of gaps')

    print('\n=== OBJECTION 2: is it just pre-market trading showing through? ===')
    for win in ['early', 'late']:
        sub = tw[tw['window'] == win]
        agg = (sub.groupby(['ticker', 'session'])
                  .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
                  .rename(columns={'session': 'date'}))
        dd = panel(agg, px)
        share = (sub.shape[0] / tw.shape[0]) * 100
        label = ('EARLY window (just after close)' if win == 'early'
                 else 'LATE window (run-up to open)')
        estimate(dd, f'{label} [{share:.0f}% of posts]')

    print('\n  The early window is the decisive one: right after the close there is')
    print('  no pre-market trading yet, so sentiment written then cannot be')
    print('  reacting to a price move the daily data hides.')


if __name__ == '__main__':
    main()
