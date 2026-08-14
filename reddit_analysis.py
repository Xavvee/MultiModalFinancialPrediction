import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from analysis.common import ci, mde, report_corr

"""Reddit against the same questions Twitter was asked.

The comparison is deliberately narrow so that any difference is attributable to
the platform: same asset (Bitcoin), same year (2022), same models, same score
definition, same tests.

Reddit brings two things Twitter did not:
  365 consecutive days, where the best Twitter corpus gave 222 scattered over 703
  a genuine engagement signal - score, comments and especially UPVOTE RATIO,
  which measures community disagreement directly and has no Twitter equivalent

One caveat colours everything: 2022 was a one-directional bear market in crypto,
so the majority-direction baseline is unusually strong and any directional
claim has to be checked against it.
"""

POSTS = 'data/reddit/processed/per_post.parquet'
PRICE_CACHE = 'data/reddit/processed/btc_2022_daily.parquet'
MIN_POSTS = 200


def load_prices():
    import os
    if os.path.exists(PRICE_CACHE):
        px = pd.read_parquet(PRICE_CACHE)
    else:
        px = yf.download('BTC-USD', start='2021-12-20', end='2023-01-05',
                         progress=False, auto_adjust=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px = px.reset_index()[['Date', 'Close', 'Volume']].rename(columns={'Date': 'date'})
        px.to_parquet(PRICE_CACHE, index=False)
    px['date'] = pd.to_datetime(px['date']).dt.tz_localize(None).dt.floor('D')
    px = px.sort_values('date').set_index('date')
    px['daily_return'] = px['Close'].pct_change()
    px['absret'] = px['daily_return'].abs()
    px['next_return'] = px['daily_return'].shift(-1)
    px['next_absret'] = px['absret'].shift(-1)
    px['prev_return'] = px['daily_return'].shift(1)
    return px


def daily_frame():
    df = pd.read_parquet(POSTS)
    df = df[df['finbert'].notna()].copy()
    d = df.groupby('date').agg(
        finbert=('finbert', 'mean'), roberta=('roberta', 'mean'),
        disp=('finbert', 'std'), n=('finbert', 'size'),
        score=('score', 'mean'), comments=('num_comments', 'mean'),
        upvote=('upvote_ratio', 'mean'))
    d = d[d['n'] >= MIN_POSTS]
    return df, d.join(load_prices(), how='inner').dropna(subset=['daily_return'])


def run():
    raw, d = daily_frame()
    print(f'Posts: {len(raw):,} | days with >= {MIN_POSTS} posts: {len(d)}')
    print(f'Range: {d.index.min().date()} -> {d.index.max().date()}')
    print(f'Median posts/day: {int(d["n"].median())}   MDE at this n: {mde(len(d)):.3f}\n')

    up = (d['daily_return'] > 0).mean()
    print(f'2022 was one-directional: {up*100:.1f}% of days up, so the majority')
    print(f'baseline already scores {max(up, 1-up)*100:.1f}% on direction.\n')

    print('=== GATE: does the measure react to price at all? ===')
    for m in ['finbert', 'roberta']:
        report_corr(d[m], d['daily_return'], f'{m} vs SAME-DAY return')

    print('\n=== THE QUESTION: does it predict tomorrow? ===')
    for m in ['finbert', 'roberta']:
        report_corr(d[m], d['next_return'], f'{m} vs NEXT-DAY return')

    print('\n=== Reddit-only signals, absent from Twitter ===')
    # upvote_ratio is a direct disagreement measure: 0.5 means the community
    # split evenly, 1.0 means near-unanimous approval.
    report_corr(d['upvote'], d['next_return'], 'mean upvote ratio -> next return')
    report_corr(d['upvote'], d['next_absret'], 'mean upvote ratio -> next |return|')
    report_corr(d['disp'], d['next_absret'], 'sentiment dispersion -> next |return|')
    report_corr(np.log(d['n']), d['next_absret'], 'log(post volume) -> next |return|')
    report_corr(d['absret'], d['next_absret'], '[reference] today |return| -> next')

    print('\n=== PLACEBO: the same measures against YESTERDAY ===')
    for m in ['finbert', 'roberta']:
        report_corr(d[m], d['prev_return'], f'{m} vs PREVIOUS-day return')

    print('\n=== Engagement-weighted aggregation ===')
    df = raw[raw['date'].isin(d.index)].copy()
    w = np.log1p(df['score'].clip(lower=0)).clip(lower=0.01)
    num = df[['finbert']].mul(w, axis=0).groupby(df['date']).sum()
    den = w.groupby(df['date']).sum()
    weighted = num.div(den, axis=0).join(load_prices(), how='inner')
    report_corr(weighted['finbert'], weighted['daily_return'], 'score-weighted, same-day')
    report_corr(weighted['finbert'], weighted['next_return'], 'score-weighted, next-day')

    print('\n=== Head-to-head with Twitter on shared days ===')
    try:
        tw = pd.read_parquet('data/new_dataset/processed/per_tweet.parquet')
        tw = tw[tw['date'].notna() & tw['finbert'].notna()]
        td = tw.groupby('date').agg(tw_finbert=('finbert', 'mean'), tw_n=('finbert', 'size'))
        td = td[(td['tw_n'] >= MIN_POSTS)]
        both = d.join(td, how='inner')
        print(f'  overlapping days: {len(both)}   MDE here: {mde(len(both)):.3f}')
        if len(both) > 30:
            r, p = stats.pearsonr(both['finbert'], both['tw_finbert'])
            print(f'  do the two platforms agree? r={r:+.4f} (p={p:.4f})')
            report_corr(both['finbert'], both['daily_return'], 'Reddit same-day (shared days)')
            report_corr(both['tw_finbert'], both['daily_return'], 'Twitter same-day (shared days)')
            report_corr(both['finbert'], both['next_return'], 'Reddit next-day (shared days)')
            report_corr(both['tw_finbert'], both['next_return'], 'Twitter next-day (shared days)')
            print('  (this block is underpowered by construction - see MDE above)')
    except FileNotFoundError:
        print('  Twitter corpus not available on this branch')


if __name__ == '__main__':
    run()
