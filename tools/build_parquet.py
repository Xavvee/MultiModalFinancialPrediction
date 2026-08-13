"""Collapse the multi-GB interim CSVs into compact per-tweet parquet files.

Everything downstream - per-account aggregation, engagement cohorts, influence
screens - needs only who posted, when, how much engagement it drew, and the two
sentiment scores. The tweet TEXT is what makes those CSVs enormous and nothing
after the NLP stage reads it, so it is dropped. The result keeps full analysis
capability at roughly 5% of the size.

Run after nlp_processor.py has produced the sentiment scores.
"""
import os
import hashlib
import pandas as pd


def hash_text(s):
    return hashlib.blake2b(str(s).encode('utf-8', 'ignore'), digest_size=8).hexdigest()


def build(tweets_csv, scores, out, user_col, extra_cols):
    """scores: a positional checkpoint CSV, or a hash-keyed cache parquet."""
    print(f'\n=== {out} ===')
    df = pd.read_csv(tweets_csv, engine='python', on_bad_lines='skip')
    print(f'  tweets: {len(df):,}')

    if scores.endswith('.csv'):
        ck = pd.read_csv(scores)
        if len(ck) != len(df):
            print(f'  !! MISALIGNED ({len(df):,} vs {len(ck):,}) - aborting.')
            print('     A positional join needs the exact same read as nlp_processor.py.')
            return
        df['finbert'] = ck['finbert_base'].values
        df['roberta'] = ck['roberta_base'].values
    else:
        cache = pd.read_parquet(scores).set_index('text_hash')
        th = df['cleaned_text'].map(hash_text)
        df['finbert'] = cache['finbert'].reindex(th).values
        df['roberta'] = cache['roberta'].reindex(th).values
        miss = df['finbert'].isna().sum()
        print(f'  cache miss: {miss:,} ({miss/len(df)*100:.2f}%)')

    df['date'] = pd.to_datetime(df['date'].astype(str).str.slice(0, 10), errors='coerce')
    df = df[df['date'].notna() & df['finbert'].notna()]

    keep = ['date', user_col] + extra_cols + ['finbert', 'roberta']
    out_df = df[keep].rename(columns={user_col: 'user'})
    out_df['user'] = out_df['user'].astype('category')
    for c in ['finbert', 'roberta', 'engagement_weight']:
        if c in out_df:
            out_df[c] = out_df[c].astype('float32')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_parquet(out, index=False, compression='zstd')
    print(f'  saved {len(out_df):,} rows -> {os.path.getsize(out)/1e6:.0f} MB')


if __name__ == '__main__':
    build('data/old_dataset/interim/weighted_tweets.csv',
          'data/old_dataset/processed/full_dataset_weighted_checkpoint.csv',
          'data/old_dataset/processed/per_tweet.parquet',
          user_col='user',
          extra_cols=['likes', 'retweets', 'replies', 'engagement_weight'])

    build('data/new_dataset/interim/weighted_tweets_2021_23.csv',
          'data/sentiment_cache.parquet',
          'data/new_dataset/processed/per_tweet.parquet',
          user_col='user_name',
          extra_cols=['user_followers', 'user_verified', 'engagement_weight'])
