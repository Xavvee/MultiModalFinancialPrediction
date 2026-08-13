import os
import math
import hashlib
import pandas as pd
import numpy as np
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

SENTIMENT_CACHE = 'data/sentiment_cache.parquet'


def hash_text(s):
    return hashlib.blake2b(str(s).encode('utf-8', 'ignore'), digest_size=8).hexdigest()


class StockNetNLP:
    """Scores stocknet tweets with the same two models used on the Bitcoin corpora.

    Shares the text-hash cache with the Bitcoin pipeline: identical text gets
    identical scores, so nothing is ever computed twice and the two corpora stay
    directly comparable.
    """

    def __init__(self, input_parquet='data/stocknet/processed/per_tweet_meta.parquet',
                 output_parquet='data/stocknet/processed/per_tweet.parquet',
                 cache_path=SENTIMENT_CACHE, batch_size=64):
        self.input_parquet = input_parquet
        self.output_parquet = output_parquet
        self.cache_path = cache_path
        self.batch_size = batch_size

        if torch_directml.is_available():
            self.device = torch_directml.device()
            print(f"Loading models on GPU: {self.device}")
        else:
            self.device = torch.device('cpu')
            print("DirectML unavailable, falling back to CPU")

        self.finbert_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_mod = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert").to(self.device).eval()
        roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.roberta_tok = AutoTokenizer.from_pretrained(roberta)
        self.roberta_mod = AutoModelForSequenceClassification.from_pretrained(
            roberta).to(self.device).eval()

    def score_batch(self, texts):
        short = [" ".join(str(t).split()[:400]) if isinstance(t, str) else "" for t in texts]
        with torch.no_grad():
            fi = self.finbert_tok(short, return_tensors='pt', padding=True,
                                  truncation=True, max_length=512)
            fi = {k: v.to(self.device) for k, v in fi.items()}
            fp = torch.nn.functional.softmax(self.finbert_mod(**fi).logits, dim=-1).cpu().numpy()
            # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
            fin = (fp[:, 0] - fp[:, 1]).tolist()

            ri = self.roberta_tok(short, return_tensors='pt', padding=True,
                                  truncation=True, max_length=512)
            ri = {k: v.to(self.device) for k, v in ri.items()}
            rp = torch.nn.functional.softmax(self.roberta_mod(**ri).logits, dim=-1).cpu().numpy()
            # cardiffnlp: 0=negative, 1=neutral, 2=positive
            rob = (rp[:, 2] - rp[:, 0]).tolist()
        return fin, rob

    def run(self):
        df = pd.read_parquet(self.input_parquet)
        print(f"Tweets: {len(df):,}")

        df['text_hash'] = df['cleaned_text'].map(hash_text)
        todo = pd.DataFrame({'text_hash': df['text_hash'].unique()})
        print(f"Unique texts: {len(todo):,} "
              f"({(1 - len(todo)/len(df))*100:.1f}% duplicates)")

        if os.path.exists(self.cache_path):
            cache = pd.read_parquet(self.cache_path)
            todo = todo.merge(cache, on='text_hash', how='left')
            hits = int(todo['finbert'].notna().sum())
            print(f"Cache hits: {hits:,} / {len(todo):,} ({hits/len(todo)*100:.1f}%)")
        else:
            cache = pd.DataFrame(columns=['text_hash', 'finbert', 'roberta'])
            todo['finbert'] = np.nan
            todo['roberta'] = np.nan

        missing = todo[todo['finbert'].isna()].copy()
        print(f"Needs inference: {len(missing):,}")

        if len(missing):
            first = df.drop_duplicates('text_hash').set_index('text_hash')['cleaned_text']
            texts = first.reindex(missing['text_hash']).tolist()
            fin, rob = [], []
            batches = math.ceil(len(texts) / self.batch_size)
            for i in tqdm(range(0, len(texts), self.batch_size), total=batches, desc="NLP"):
                f, r = self.score_batch(texts[i:i + self.batch_size])
                fin.extend(f)
                rob.extend(r)
            missing['finbert'] = fin
            missing['roberta'] = rob

            cache = pd.concat([cache, missing[['text_hash', 'finbert', 'roberta']]],
                              ignore_index=True).drop_duplicates('text_hash')
            cache.to_parquet(self.cache_path, index=False)
            print(f"Cache updated -> {len(cache):,} entries")

            todo = todo.set_index('text_hash')
            todo.loc[missing['text_hash'], ['finbert', 'roberta']] = \
                missing[['finbert', 'roberta']].values
            todo = todo.reset_index()

        scores = todo.set_index('text_hash')
        df['finbert'] = scores['finbert'].reindex(df['text_hash']).values
        df['roberta'] = scores['roberta'].reindex(df['text_hash']).values
        df = df.drop(columns=['cleaned_text', 'text_hash'])

        os.makedirs(os.path.dirname(self.output_parquet), exist_ok=True)
        df.to_parquet(self.output_parquet, index=False, compression='zstd')
        print(f"Saved {len(df):,} scored tweets -> {self.output_parquet}")
        print(f"  finbert mean {df['finbert'].mean():+.4f} | roberta mean {df['roberta'].mean():+.4f}")


if __name__ == "__main__":
    StockNetNLP().run()
