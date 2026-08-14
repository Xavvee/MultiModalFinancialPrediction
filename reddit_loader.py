import os
import re
import glob
import numpy as np
import pandas as pd

"""ETL for the Reddit crypto corpus (2022).

Reddit differs from Twitter in ways that matter for this comparison:

  LONGER TEXT      a post carries a title AND a body, so there is far more to
                   read than in 280 characters. Both are scored together.

  REAL ENGAGEMENT  score, comment count and upvote ratio are community votes,
                   not vanity metrics inflated by automated accounts. The
                   upvote ratio in particular has no Twitter equivalent - it
                   measures disagreement directly.

  CONTINUOUS       365 consecutive days, where the best Twitter corpus offered
                   222 days scattered across 703.

Output mirrors the Twitter per_tweet parquet so the existing analysis applies.
"""

RAW_DIR = 'data/reddit/raw'
OUT = 'data/reddit/processed/per_post_meta.parquet'

# Removed/deleted posts keep their metadata but lose their text; scoring the
# placeholder would inject thousands of identical meaningless rows.
DEAD_TEXT = {'[removed]', '[deleted]', '', 'nan'}


def clean_text(text):
    """Same normalisation as the Twitter pipelines, so scores stay comparable
    and the shared sentiment cache can be reused across corpora."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'/?u/\w+', '', text)      # Reddit's @-mention equivalent
    text = re.sub(r'/?r/\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run(raw_dir=RAW_DIR, out=OUT):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.csv'))):
        sub = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, engine='python', on_bad_lines='skip',
                         usecols=['author', 'created', 'title', 'selftext',
                                  'score', 'num_comments', 'upvote_ratio', 'removed',
                                  'deleted'])
        df['subreddit'] = sub
        frames.append(df)
        print(f'  {sub:16s} {len(df):>8,} posts')

    df = pd.concat(frames, ignore_index=True)
    print(f'\nTotal read: {len(df):,}')

    df['ts'] = pd.to_datetime(df['created'], unit='s', errors='coerce')
    df = df[df['ts'].notna()].copy()
    df['date'] = df['ts'].dt.floor('D')

    title = df['title'].fillna('').astype(str)
    body = df['selftext'].fillna('').astype(str)
    body = body.where(~body.str.strip().str.lower().isin(DEAD_TEXT), '')
    # Title and body are scored as one document: the title alone is often a bare
    # headline, and the body alone is often empty for link posts.
    df['cleaned_text'] = (title + '. ' + body).map(clean_text)
    df = df[df['cleaned_text'].str.len() >= 10].copy()

    for c in ['score', 'num_comments', 'upvote_ratio']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['score'] = df['score'].fillna(0)
    df['num_comments'] = df['num_comments'].fillna(0)

    # Same shape as the Twitter engagement weight so cohort code transfers.
    df['engagement_weight'] = np.log1p(df['score'].clip(lower=0)
                                       + 20 * df['num_comments'].clip(lower=0))

    keep = ['ts', 'date', 'subreddit', 'author', 'cleaned_text',
            'score', 'num_comments', 'upvote_ratio', 'engagement_weight']
    out_df = df[keep].rename(columns={'author': 'user'})
    out_df['user'] = out_df['user'].astype('category')
    out_df['subreddit'] = out_df['subreddit'].astype('category')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_parquet(out, index=False, compression='zstd')

    print(f'\nKept {len(out_df):,} posts with usable text')
    print(f'  range: {out_df["ts"].min()} -> {out_df["ts"].max()}')
    print(f'  days covered: {out_df["date"].nunique()}')
    print(f'  authors: {out_df["user"].nunique():,}')
    print(f'  median posts/day: {int(out_df.groupby("date").size().median())}')
    print(f'  median text length: {int(out_df["cleaned_text"].str.len().median())} chars')
    print(f'  saved -> {out} ({os.path.getsize(out)/1e6:.0f} MB)')
    return out_df


if __name__ == '__main__':
    run()
