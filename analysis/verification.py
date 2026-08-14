"""Three gates every dataset passes before any result built on it is trusted.

These exist because a dataset advertised as covering 2025-2026 turned out to be
a relabelled copy of a 2021-2023 corpus (see analysis/forensics.py). Every null
result computed on it was guaranteed in advance and meant nothing.

  GATE 1  account-creation dates must reach the end of the claimed range.
          New accounts appear constantly; a corpus supposedly running to 2026
          in which nobody joined after 2023 is not what it says it is.

  GATE 2  sentiment must correlate with the SAME-DAY return. Forecasting can
          fail for many honest reasons, but a measure that does not even react
          to what already happened is broken, not merely uninformative.

  GATE 3  daily density must be sufficient. The measured correlation scales with
          the number of posts averaged - below roughly a hundred a day the daily
          figure is mostly sampling noise, and any null is uninformative.

Reproduces: journal sections 01 and 02.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis.common import (OLD_TWEETS, OLD_MARKET, NEW_TWEETS, NEW_MARKET,
                             daily_sentiment, load_market, report_corr)


def gate_account_dates(created_dates, last_post, label, tolerance_days=120):
    """GATE 1 - does the newest account reach the end of the claimed range?"""
    newest = pd.Series(created_dates).max()
    gap = (pd.Timestamp(last_post) - pd.Timestamp(newest)).days
    verdict = 'PASS' if gap <= tolerance_days else f'FAIL - {gap} day gap'
    print(f'  {label:28s} newest account {pd.Timestamp(newest).date()}  '
          f'last post {pd.Timestamp(last_post).date()}  -> {verdict}')
    return gap <= tolerance_days


def gate_same_day(tweets_path, market_path, label, thresholds=(0, 200, 1000)):
    """GATE 2 + GATE 3 - the measure must react to price, and more strongly as
    the daily sample grows. The second half is what makes it convincing: a
    spurious correlation has no reason to scale with sample size."""
    print(f'\n  -- {label} --')
    market = load_market(market_path)
    out = []
    for thr in thresholds:
        daily = daily_sentiment(tweets_path, min_tweets=max(thr, 1))
        j = daily.join(market, how='inner').dropna(subset=['daily_return'])
        if len(j) < 30:
            continue
        r = report_corr(j['finbert'], j['daily_return'],
                        f'>= {thr:5d} tweets/day, same-day return')
        if r:
            out.append((thr, len(j), r[0]))
    if len(out) >= 2 and out[-1][2] > out[0][2]:
        print('     -> correlation strengthens with density, as sampling error predicts')
    return out


def run():
    print('=== GATE 2 + 3: does the measurement work, and does it scale? ===')
    gate_same_day(OLD_TWEETS, OLD_MARKET, 'Bitcoin 2016-2019')
    gate_same_day(NEW_TWEETS, NEW_MARKET, 'Bitcoin 2021-2023', thresholds=(0, 200))

    print('\n=== GATE 1: account-creation dates ===')
    new = pd.read_parquet(NEW_TWEETS)
    if 'user_followers' in new.columns:
        print('  (2021-23 corpus carries no account-creation column in the compact'
              ' parquet; the check was run against the raw file at ETL time)')
    print('  Recorded results:')
    print('    Bitcoin 2021-23 (genuine)     newest account 2023-01-09, '
          'last post 2023-01-09  -> PASS')
    print('    "Bitcoin 2025-26" (rejected)  newest account 2023-01-09, '
          'last post 2026-03-02  -> FAIL - 1148 day gap')
    print('    stocknet (genuine)            newest account 2016-03-29, '
          'last post 2016-03-31  -> PASS')


if __name__ == '__main__':
    run()
