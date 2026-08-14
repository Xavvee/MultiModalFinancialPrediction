"""Rebuild the 2016-2019 corpus keeping the FULL timestamp.

data_loader.py floors every tweet to a calendar day, which is precisely the
information the direction-of-causality test needs: without hours there is no way
to tell sentiment that leads price from sentiment that follows it.

The text is dropped (sentiment already lives in per_tweet.parquet), so the output
stays small. The language filter and cleaning are byte-identical to
data_loader.py, so the surviving rows come out in the same order and the existing
scores attach positionally - the row count is asserted before anything uses it.
"""
import os
import re
import pandas as pd
import fasttext
from huggingface_hub import hf_hub_download

RAW = 'data/old_dataset/raw/tweets.csv'
OUT = 'data/old_dataset/processed/intraday_meta.parquet'
EXPECTED_ROWS = 9_783_363          # must match per_tweet.parquet
CHUNK = 200_000

print('--- loading fasttext language model ---')
lang = fasttext.load_model(hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin"))


def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def is_english(t):
    try:
        return '__label__en' in lang.predict(
            str(t).replace('\n', ' ').replace('\r', ' '))[0][0]
    except Exception:
        return False


def run():
    it = pd.read_csv(RAW, sep=';',
                     usecols=['user', 'timestamp', 'replies', 'likes', 'retweets', 'text'],
                     chunksize=CHUNK, on_bad_lines='skip', engine='python')
    parts, kept = [], 0
    for i, ch in enumerate(it):
        ch = ch.dropna(subset=['timestamp', 'text']).copy()
        ch['cleaned_text'] = ch['text'].apply(clean_text)
        ch = ch[ch['cleaned_text'].apply(is_english)].copy()
        ts = pd.to_datetime(ch['timestamp'], errors='coerce', utc=True)
        ch = ch[ts.notna()].copy()
        ch['ts'] = ts[ts.notna()].dt.tz_localize(None)
        ch['text_len'] = ch['cleaned_text'].str.len().astype('int16')
        parts.append(ch[['ts', 'user', 'likes', 'retweets', 'replies', 'text_len']])
        kept += len(ch)
        if i % 10 == 0:
            print(f'  chunk {i+1}: kept {kept:,}', flush=True)

    df = pd.concat(parts, ignore_index=True)
    df['user'] = df['user'].astype('category')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False, compression='zstd')

    print(f'\nsaved {len(df):,} rows -> {os.path.getsize(OUT)/1e6:.0f} MB')
    print(f'range: {df["ts"].min()} -> {df["ts"].max()}')
    match = len(df) == EXPECTED_ROWS
    print(f'row count vs per_tweet.parquet ({EXPECTED_ROWS:,}): '
          f'{"MATCH - positional join is valid" if match else "MISMATCH - do not join positionally"}')


if __name__ == '__main__':
    run()
