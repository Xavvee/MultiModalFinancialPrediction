import os
import numpy as np
import pandas as pd
from scipy import stats

"""Positive control: can this pipeline detect an effect that is known to exist?

Every null result so far invites the same objection - maybe the method simply
cannot detect sentiment effects at all. The Musk/Dogecoin episode of 2021 is the
cleanest available answer: an effect that is documented, large, and attributable
to identifiable posts.

The design has a control group built in. Musk's tweets that say nothing about
crypto share the author, the posting hours and the audience, but carry no
Dogecoin content - so any price move following them is the baseline that
DOGE-related tweets must beat.

Two levels, and the second is the one that matters:
  1. EVENT   do DOGE tweets move the price? (validates the data and the design)
  2. SIGNAL  does OUR sentiment score of those tweets track the move?
             (validates the measurement this thesis rests on)

A placebo window before each tweet guards against reverse causality - Musk may
well be tweeting because the price is already moving.
"""

TWEETS = 'data/dogecoin/raw/musk_tweets_2021.csv'
PRICES = 'data/dogecoin/raw/doge_prices_2021.csv'
SENT_OUT = 'data/dogecoin/processed/musk_tweets_scored.parquet'
WINDOWS = [5, 15, 30, 60, 240]


def load():
    tw = pd.read_csv(TWEETS)
    tw['ts'] = pd.to_datetime(tw['Datetime'], format='%d/%m/%Y %H:%M', errors='coerce')
    tw = tw[tw['ts'].notna()].copy()
    tw['text'] = tw['Text'].astype(str)
    low = tw['text'].str.lower()
    # "doge" also catches "dogecoin"; the shiba/crypto terms are deliberately
    # excluded from the treatment group to keep it unambiguous.
    tw['is_doge'] = low.str.contains('doge', na=False)

    px = pd.read_csv(PRICES)
    px['ts'] = pd.to_datetime(px['open_time'], format='%d/%m/%Y %H:%M', errors='coerce')
    px = px[px['ts'].notna()].drop_duplicates('ts').set_index('ts').sort_index()
    return tw, px['price']


def window_return(price, t, minutes, forward=True):
    """Return over `minutes` starting (or ending) at t, using the nearest print."""
    a, b = (t, t + pd.Timedelta(minutes=minutes)) if forward else \
           (t - pd.Timedelta(minutes=minutes), t)
    idx = price.index
    ia, ib = idx.searchsorted(a), idx.searchsorted(b)
    if ia >= len(idx) or ib >= len(idx):
        return np.nan
    pa, pb = price.iloc[ia], price.iloc[ib]
    if not np.isfinite(pa) or not np.isfinite(pb) or pa <= 0:
        return np.nan
    return pb / pa - 1


def compare(df, col, label):
    a = df.loc[df['is_doge'], col].dropna()
    b = df.loc[~df['is_doge'], col].dropna()
    if len(a) < 10 or len(b) < 10:
        return
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f'  {label:28s} DOGE: {a.mean()*100:+.3f}% (n={len(a)})   '
          f'kontrola: {b.mean()*100:+.3f}% (n={len(b)})   '
          f'różnica {(a.mean()-b.mean())*100:+.3f} p.p.  t={t:+5.2f} p={p:.4f}'
          f'{"  ***" if p < 0.05 else ""}')


def main():
    tw, price = load()
    print(f'Tweets: {len(tw):,}   DOGE-related: {int(tw["is_doge"].sum())}   '
          f'control: {int((~tw["is_doge"]).sum())}')
    print(f'Prices: {len(price):,} minute bars\n')

    for w in WINDOWS:
        tw[f'fwd{w}'] = [window_return(price, t, w, True) for t in tw['ts']]
    for w in [60]:
        tw[f'pre{w}'] = [window_return(price, t, w, False) for t in tw['ts']]

    print('=== 1. EVENT STUDY: price move AFTER the tweet ===')
    for w in WINDOWS:
        compare(tw, f'fwd{w}', f'+{w} min')

    print('\n=== PLACEBO: price move BEFORE the tweet (reverse causality) ===')
    compare(tw, 'pre60', '-60 min')

    # ---------------- level 2: does OUR sentiment measure track it? ----------
    print('\n=== 2. DOES OUR SENTIMENT MEASURE PICK IT UP? ===')
    if os.path.exists(SENT_OUT):
        scored = pd.read_parquet(SENT_OUT)
        tw = tw.reset_index(drop=True)
        tw['finbert'] = scored['finbert'].values
        tw['roberta'] = scored['roberta'].values
    else:
        print('  (run dogecoin_sentiment.py first)')
        return

    doge = tw[tw['is_doge']].copy()
    print(f'  DOGE tweets scored: {len(doge)}')
    for m in ['finbert', 'roberta']:
        print(f'  -- {m} --')
        for w in [15, 60, 240]:
            s = doge[doge[f'fwd{w}'].notna()]
            if len(s) < 20:
                continue
            r, p = stats.pearsonr(s[m], s[f'fwd{w}'])
            print(f'     sentiment -> return +{w:3d} min: r={r:+.4f} (p={p:.4f}, n={len(s)})'
                  f'{"  SIGNIFICANT" if p < 0.05 else ""}')
        s = doge[doge['pre60'].notna()]
        r, p = stats.pearsonr(s[m], s['pre60'])
        print(f'     PLACEBO: sentiment -> return -60 min: r={r:+.4f} (p={p:.4f})')

    print('\n=== 3. Positive vs negative DOGE tweets ===')
    for m in ['finbert', 'roberta']:
        pos = doge.loc[doge[m] > doge[m].median(), 'fwd60'].dropna()
        neg = doge.loc[doge[m] <= doge[m].median(), 'fwd60'].dropna()
        t, p = stats.ttest_ind(pos, neg, equal_var=False)
        print(f'  {m:8s} +60min after upbeat: {pos.mean()*100:+.3f}%  '
              f'after downbeat: {neg.mean()*100:+.3f}%   '
              f't={t:+5.2f} p={p:.4f}{"  ***" if p < 0.05 else ""}')

    print('\n=== 4. The single largest moves, for context ===')
    top = doge.reindex(doge['fwd60'].abs().sort_values(ascending=False).index).head(5)
    for _, r in top.iterrows():
        print(f'  {r["ts"]}  {r["fwd60"]*100:+7.2f}% in 60min  '
              f'finbert={r["finbert"]:+.2f}  "{r["text"][:60]}..."')


if __name__ == '__main__':
    main()
