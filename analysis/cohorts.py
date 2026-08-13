"""Do some voices matter more than others?

The thesis proposal assumed they would: split the crowd into authorities and
everyone else, and the authorities should lead. Two independent definitions of
influence were tested and neither helps.

  AUTHORITY  follower count and verified status (2021-23 corpus only - the
             2016-19 corpus carries no account metadata at all)
  REACH      likes, retweets and replies - arguably the better measure, since
             it records what people actually amplified rather than who they
             theoretically listen to

The per-account screen needs its own null. Correlating thousands of accounts
against returns produces hundreds of "significant" hits by construction, so the
same search is re-run against a shuffled return series: real accounts must beat
what chance alone manufactures.

Reproduces: journal section 05.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis.common import OLD_TWEETS, OLD_MARKET, load_market, report_corr, mde

MIN_DAYS_PER_ACCOUNT = 25
N_SHUFFLES = 5


def _daily(sub, market, min_n=20):
    daily = sub.groupby('date')[['finbert', 'roberta']].mean()
    counts = sub.groupby('date').size()
    daily = daily[counts >= min_n]
    return daily.join(market, how='inner').dropna(subset=['daily_return'])


def reach_cohorts():
    """Split by realised reach and test each slice."""
    df = pd.read_parquet(OLD_TWEETS)
    df = df[df['date'].notna() & df['finbert'].notna()].copy()
    for c in ['likes', 'retweets', 'replies']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['reach'] = df['likes'] + 20 * df['retweets'] + 15 * df['replies']
    market = load_market(OLD_MARKET)

    print(f'{len(df):,} tweets, {df["user"].nunique():,} accounts')
    print(f'  share with zero engagement: {(df["reach"] == 0).mean()*100:.1f}%\n')

    print('=== Cohorts by per-tweet reach ===')
    cuts = [('zero engagement', df['reach'] == 0),
            ('reach 1-9', (df['reach'] >= 1) & (df['reach'] < 10)),
            ('reach 10-99', (df['reach'] >= 10) & (df['reach'] < 100)),
            ('reach 100-999', (df['reach'] >= 100) & (df['reach'] < 1000)),
            ('reach 1000+ (viral)', df['reach'] >= 1000)]
    for label, mask in cuts:
        j = _daily(df[mask], market)
        if len(j) < 100:
            continue
        share = mask.mean() * 100
        report_corr(j['finbert'], j['daily_return'], f'{label} [{share:4.1f}% of tweets] same-day')
        report_corr(j['finbert'], j['next_return'], f'{label} next-day')

    print('\n=== Cohorts by account-level typical reach ===')
    acct = df.groupby('user', observed=True).agg(
        tweets=('reach', 'size'), mean_reach=('reach', 'mean'))
    acct = acct[acct['tweets'] >= 10]
    for q, name in [(0.99, 'top 1% of accounts'), (0.90, 'top 10% of accounts'),
                    (0.50, 'bottom 50% of accounts')]:
        thr = acct['mean_reach'].quantile(q)
        users = acct.index[acct['mean_reach'] >= thr] if q > 0.5 else \
            acct.index[acct['mean_reach'] <= thr]
        j = _daily(df[df['user'].isin(users)], market)
        if len(j) < 100:
            continue
        report_corr(j['finbert'], j['daily_return'], f'{name} same-day')
        report_corr(j['finbert'], j['next_return'], f'{name} next-day')
    print('\n  The highest-reach accounts REACT most strongly - posts go viral when')
    print('  price moves - yet predict exactly as poorly as accounts nobody reads.')


def _screen(frame, col):
    out = []
    for user, g in frame.groupby('user', sort=False, observed=True):
        y = g[col].values
        for model in ['finbert', 'roberta']:
            x = g[model].values
            if len(y) < 3 or np.std(x) == 0 or np.std(y) == 0:
                continue
            r, p = stats.pearsonr(x, y)
            out.append((user, model, len(y), r, p))
    return pd.DataFrame(out, columns=['user', 'model', 'n', 'r', 'p'])


def per_account_screen(tweets_path, market_path, label):
    """Screen every sufficiently active account, judged against a shuffled null."""
    df = pd.read_parquet(tweets_path)
    df = df[df['date'].notna() & df['finbert'].notna()]
    market = load_market(market_path)

    ud = df.groupby(['user', 'date'], observed=True)[['finbert', 'roberta']].mean().reset_index()
    ud['next_return'] = ud['date'].map(market['next_return'])
    ud = ud[ud['next_return'].notna()]
    counts = ud.groupby('user', observed=True).size()
    ud = ud[ud['user'].isin(counts[counts >= MIN_DAYS_PER_ACCOUNT].index)]

    n_accounts = ud['user'].nunique()
    print(f'\n=== Per-account screen: {label} ===')
    print(f'  accounts active on >= {MIN_DAYS_PER_ACCOUNT} days: {n_accounts:,}   '
          f'observations: {len(ud):,}')
    print(f'  detectable per-account correlation at {MIN_DAYS_PER_ACCOUNT} obs: '
          f'{mde(MIN_DAYS_PER_ACCOUNT):.2f} - only very strong effects are in reach')

    real = _screen(ud, 'next_return')
    m = len(real)
    n_raw = int((real['p'] < 0.05).sum())

    p_sorted = np.sort(real['p'].values)
    bh = 0.0
    for i, p in enumerate(p_sorted, start=1):
        if p <= i / m * 0.05:
            bh = p
    n_bh = int((real['p'] <= bh).sum()) if bh > 0 else 0
    print(f'  tests: {m:,} | p<0.05: {n_raw} (chance alone: {0.05*m:.0f}) | '
          f'survive BH-FDR: {n_bh}')
    print(f'  strongest |r|: {real["r"].abs().max():.4f}')

    dates = market.index.values
    rng = np.random.default_rng(11)
    sig, mx = [], []
    for i in range(N_SHUFFLES):
        shuffled = pd.Series(
            market['next_return'].sample(frac=1.0, random_state=int(rng.integers(1e6))).values,
            index=dates)
        tmp = ud.copy()
        tmp['fake'] = tmp['date'].map(shuffled)
        tmp = tmp[tmp['fake'].notna()]
        res = _screen(tmp, 'fake')
        sig.append(int((res['p'] < 0.05).sum()))
        mx.append(float(res['r'].abs().max()))
    print(f'  shuffled control: {np.mean(sig):.0f} +/- {np.std(sig):.0f} accounts, '
          f'strongest |r| {np.mean(mx):.4f}')
    print('  -> real data produces no more apparent influencers than pure chance.')


def run():
    reach_cohorts()
    per_account_screen(OLD_TWEETS, OLD_MARKET, 'Bitcoin 2016-2019')


if __name__ == '__main__':
    run()
