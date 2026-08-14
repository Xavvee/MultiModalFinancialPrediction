import glob
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TWEETS = 'data/stocknet/processed/per_tweet.parquet'
PRICES = 'data/stocknet/price/raw'

# NYSE is shut outside roughly 13:30-21:00 UTC (the window shifts an hour with
# US daylight saving). Taking 21:00 -> 13:30 as "closed" is conservative: it is
# shut across that whole span under either regime, so sentiment measured inside
# it genuinely cannot have been traded on yet.
CLOSE_UTC = 21
OPEN_UTC = 13.5


def load_prices():
    frames = []
    for path in sorted(glob.glob(os.path.join(PRICES, '*.csv'))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['Date'])
        df['ticker'] = ticker
        frames.append(df[['ticker', 'date', 'Open', 'Close', 'Adj Close', 'Volume']])
    px = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'date'])
    px['prev_close'] = px.groupby('ticker')['Close'].shift(1)
    # Overnight gap: what the market does at the first print after being shut.
    px['gap'] = px['Open'] / px['prev_close'] - 1
    # Intraday: open to close, the part that CAN respond to overnight news.
    px['intraday'] = px['Close'] / px['Open'] - 1
    px['daily_return'] = px['Close'] / px['prev_close'] - 1
    px['next_return'] = px.groupby('ticker')['daily_return'].shift(-1)
    return px


def assign_session(ts):
    """Map each tweet to the trading day whose OPEN it could first affect."""
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day = ts.dt.floor('D')
    # After the close, the next session is the following calendar day;
    # before the open, it is the same day.
    after_close = hour >= CLOSE_UTC
    session = day.where(~after_close, day + pd.Timedelta(days=1))
    closed = after_close | (hour < OPEN_UTC)
    return session, closed


def main():
    tw = pd.read_parquet(TWEETS)
    tw = tw[tw['finbert'].notna()].copy()
    tw['ts'] = pd.to_datetime(tw['ts'])
    print(f"Tweets: {len(tw):,} | companies: {tw['ticker'].nunique()} | "
          f"{tw['ts'].min().date()} -> {tw['ts'].max().date()}")

    px = load_prices()
    tw['session'], tw['closed'] = assign_session(tw['ts'])

    # ---------------- GATE 2: does the measurement work here at all? ----------
    print("\n=== GATE: sentiment vs SAME-DAY return (measurement sanity) ===")
    daily = (tw.groupby(['ticker', tw['ts'].dt.floor('D')])
               .agg(sent=('finbert', 'mean'), n=('finbert', 'size')).reset_index()
               .rename(columns={'ts': 'date'}))
    m = daily.merge(px, on=['ticker', 'date'], how='inner')
    for thr in [1, 5, 10, 20]:
        s = m[(m['n'] >= thr) & m['daily_return'].notna()]
        if len(s) < 50:
            continue
        r, p = stats.pearsonr(s['sent'], s['daily_return'])
        flag = '  PASS' if (r > 0 and p < 0.05) else ''
        print(f"  company-days with >={thr:2d} tweets: n={len(s):6,}  r={r:+.4f} (p={p:.2e}){flag}")

    # ---------------- MAIN TEST: overnight sentiment -> opening gap -----------
    print("\n=== MAIN TEST: sentiment while the market is SHUT -> opening gap ===")
    print("    (the window stocks offer and 24/7 crypto does not)")
    night = tw[tw['closed']]
    agg = (night.groupby(['ticker', 'session'])
                .agg(sent=('finbert', 'mean'), sent_r=('roberta', 'mean'),
                     disp=('finbert', 'std'), n=('finbert', 'size')).reset_index()
                .rename(columns={'session': 'date'}))
    j = agg.merge(px, on=['ticker', 'date'], how='inner')
    print(f"  company-sessions with overnight tweets: {len(j):,}")

    for thr in [1, 3, 5, 10]:
        s = j[(j['n'] >= thr) & j['gap'].notna()]
        if len(s) < 50:
            continue
        r, p = stats.pearsonr(s['sent'], s['gap'])
        flag = '  <-- SIGNIFICANT' if p < 0.05 else ''
        print(f"  >={thr:2d} tweets: n={len(s):6,}  overnight sent -> GAP        "
              f"r={r:+.4f} (p={p:.4f}){flag}")

    # Panel with company fixed effects, clustered by date
    s = j[(j['n'] >= 3) & j['gap'].notna()].copy()
    s['sent_z'] = (s['sent'] - s['sent'].mean()) / s['sent'].std()
    X = pd.get_dummies(s['ticker'], prefix='t', drop_first=True).astype(float)
    X['sent_z'] = s['sent_z'].values
    X['log_n'] = np.log(s['n'].values)
    X = sm.add_constant(X)
    fit = sm.OLS(s['gap'].values, X).fit(
        cov_type='cluster', cov_kwds={'groups': s['date'].factorize()[0]})
    c, se, p = fit.params['sent_z'], fit.bse['sent_z'], fit.pvalues['sent_z']
    print(f"\n  Panel (company fixed effects, SE clustered by date, n={int(fit.nobs):,}):")
    print(f"    overnight sentiment -> gap: coef={c:+.6f} t={c/se:+.2f} p={p:.4f}"
          f"{'  <-- SIGNIFICANT' if p < 0.05 else ''}")

    # ---------------- PLACEBO: the same sentiment vs what already happened ----
    print("\n=== PLACEBO: overnight sentiment vs the PREVIOUS session's return ===")
    j2 = j.copy()
    j2['prev_return'] = j2.groupby('ticker')['daily_return'].shift(1)
    s2 = j2[(j2['n'] >= 3) & j2['prev_return'].notna()]
    r, p = stats.pearsonr(s2['sent'], s2['prev_return'])
    print(f"  n={len(s2):,}  r={r:+.4f} (p={p:.4f})"
          f"{'  <-- reacts to the past, as expected' if p < 0.05 else ''}")

    # ---------------- MARKET LEVEL: pool every company ------------------------
    print("\n=== MARKET LEVEL: all 87 companies pooled per session ===")
    mkt = (night.groupby('session').agg(sent=('finbert', 'mean'), n=('finbert', 'size'))
                .rename_axis('date').reset_index())
    mkt_px = (px.groupby('date').agg(gap=('gap', 'mean'), ret=('daily_return', 'mean'),
                                     nxt=('next_return', 'mean')).reset_index())
    mm = mkt.merge(mkt_px, on='date', how='inner')
    mm = mm[mm['n'] >= 20]
    print(f"  sessions: {len(mm):,}  (median {int(mm['n'].median())} tweets/session)")
    for label, col in [('-> market GAP', 'gap'), ('-> market same-day', 'ret'),
                       ('-> market next-day', 'nxt')]:
        s3 = mm[mm[col].notna()]
        r, p = stats.pearsonr(s3['sent'], s3[col])
        print(f"    market sentiment {label:22s} r={r:+.4f} (p={p:.4f})"
              f"{'  <-- SIGNIFICANT' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
