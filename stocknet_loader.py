import json
import os
import re
import glob
import datetime as dt
import pandas as pd
import numpy as np


class StockNetLoader:
    """ETL for the stocknet corpus (Xu & Cohen, ACL 2018).

    Structurally different from the Bitcoin corpora: instead of one flat CSV it
    ships one JSON-lines file per (company, day), and every line is a full
    Twitter API object. That object carries follower counts and verified status,
    which the 2016-19 Bitcoin corpus lacks entirely - so authority cohorts can be
    reconstructed here.

    The output mirrors the Bitcoin per_tweet parquet (date, user, engagement,
    sentiment-ready text hash) so the existing analysis code applies unchanged.
    """

    def __init__(self, tweet_dir='data/stocknet/tweet/raw',
                 output_file='data/stocknet/processed/per_tweet_meta.parquet'):
        self.tweet_dir = tweet_dir
        self.output_file = output_file

    @staticmethod
    def clean_text(text):
        """Same normalisation as the Bitcoin pipeline, so sentiment scores are
        comparable across corpora and the shared cache can be reused."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def parse_twitter_ts(value):
        try:
            return dt.datetime.strptime(value, '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
        except Exception:
            return None

    def process_and_save(self):
        files = sorted(glob.glob(os.path.join(self.tweet_dir, '*', '*')))
        print(f"--- STOCKNET ETL: {len(files):,} company-day files ---")

        rows = []
        skipped = 0
        for i, path in enumerate(files):
            ticker = os.path.basename(os.path.dirname(path))
            with open(path, encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    try:
                        tw = json.loads(line)
                    except Exception:
                        skipped += 1
                        continue

                    user = tw.get('user') or {}
                    ts = self.parse_twitter_ts(tw.get('created_at', ''))
                    if ts is None:
                        skipped += 1
                        continue

                    # 'text' is the raw tweet; the preprocessed folder ships a
                    # tokenised version, but we re-clean the raw one so the text
                    # matches what the Bitcoin pipeline fed to the models.
                    cleaned = self.clean_text(tw.get('text', ''))
                    if not cleaned:
                        skipped += 1
                        continue

                    rows.append((
                        ticker,
                        ts,
                        user.get('screen_name') or '',
                        user.get('followers_count') or 0,
                        bool(user.get('verified')),
                        tw.get('retweet_count') or 0,
                        tw.get('favorite_count') or 0,
                        cleaned,
                    ))

            if (i + 1) % 5000 == 0:
                print(f"   {i+1:,}/{len(files):,} files, {len(rows):,} tweets", flush=True)

        df = pd.DataFrame(rows, columns=[
            'ticker', 'ts', 'user', 'user_followers', 'user_verified',
            'retweets', 'likes', 'cleaned_text'])
        df['date'] = df['ts'].dt.floor('D')

        # Engagement weight uses the same formula as the Bitcoin pipeline so the
        # cohort logic transfers. stocknet carries no reply counts, so that term
        # is absent rather than imputed.
        df['engagement_weight'] = np.log1p(df['likes'] + 20 * df['retweets'])

        df['ticker'] = df['ticker'].astype('category')
        df['user'] = df['user'].astype('category')

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df.to_parquet(self.output_file, index=False, compression='zstd')

        print("\n" + "=" * 50)
        print("--- STOCKNET ETL COMPLETE ---")
        print(f"Tweets kept: {len(df):,}   (skipped {skipped:,})")
        print(f"Companies: {df['ticker'].nunique()}   Accounts: {df['user'].nunique():,}")
        print(f"Range: {df['ts'].min()} -> {df['ts'].max()}")
        print(f"Saved to {self.output_file} "
              f"({os.path.getsize(self.output_file)/1e6:.0f} MB)")
        print("=" * 50)
        return df


if __name__ == "__main__":
    loader = StockNetLoader()
    loader.process_and_save()
