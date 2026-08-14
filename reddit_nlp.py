import os
import math
import hashlib
import pandas as pd
import numpy as np
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

"""Scores Reddit posts with the same two models used on every other corpus.

Identical cleaning, identical models, identical score definition - otherwise a
platform comparison would confound the platform with the measurement. The shared
text-hash cache means anything already scored is never recomputed.
"""

CACHE = 'data/sentiment_cache.parquet'
IN = 'data/reddit/processed/per_post_meta.parquet'
OUT = 'data/reddit/processed/per_post.parquet'


def hash_text(s):
    return hashlib.blake2b(str(s).encode('utf-8', 'ignore'), digest_size=8).hexdigest()


class RedditNLP:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        if torch_directml.is_available():
            self.device = torch_directml.device()
            print(f'GPU: {self.device}')
        else:
            self.device = torch.device('cpu')
            print('DirectML unavailable, using CPU')

        self.f_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.f_mod = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert").to(self.device).eval()
        r = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.r_tok = AutoTokenizer.from_pretrained(r)
        self.r_mod = AutoModelForSequenceClassification.from_pretrained(
            r).to(self.device).eval()

    def score(self, texts):
        # 256 tokens covers the overwhelming majority of posts; the tail of very
        # long essays contributes little beyond its opening and costs quadratic
        # attention time. Twitter posts never approached this limit, so the
        # scores stay comparable across corpora.
        short = [" ".join(str(t).split()[:200]) if isinstance(t, str) else "" for t in texts]
        with torch.no_grad():
            fi = self.f_tok(short, return_tensors='pt', padding=True,
                            truncation=True, max_length=256)
            fi = {k: v.to(self.device) for k, v in fi.items()}
            fp = torch.nn.functional.softmax(self.f_mod(**fi).logits, dim=-1).cpu().numpy()
            ri = self.r_tok(short, return_tensors='pt', padding=True,
                            truncation=True, max_length=256)
            ri = {k: v.to(self.device) for k, v in ri.items()}
            rp = torch.nn.functional.softmax(self.r_mod(**ri).logits, dim=-1).cpu().numpy()
        # ProsusAI: 0=positive 1=negative | cardiffnlp: 0=negative 2=positive
        return (fp[:, 0] - fp[:, 1]).tolist(), (rp[:, 2] - rp[:, 0]).tolist()

    def run(self):
        df = pd.read_parquet(IN)
        print(f'Posts: {len(df):,}')
        df['text_hash'] = df['cleaned_text'].map(hash_text)

        todo = pd.DataFrame({'text_hash': df['text_hash'].unique()})
        print(f'Unique texts: {len(todo):,} '
              f'({(1 - len(todo)/len(df))*100:.1f}% duplicates)')

        if os.path.exists(CACHE):
            cache = pd.read_parquet(CACHE)
            todo = todo.merge(cache, on='text_hash', how='left')
            hits = int(todo['finbert'].notna().sum())
            print(f'Cache: {len(cache):,} entries -> {hits:,} hits '
                  f'({hits/max(len(todo),1)*100:.1f}%)')
        else:
            cache = pd.DataFrame(columns=['text_hash', 'finbert', 'roberta'])
            todo['finbert'] = np.nan
            todo['roberta'] = np.nan

        missing = todo[todo['finbert'].isna()].copy()
        print(f'Needs inference: {len(missing):,}')

        if len(missing):
            first = df.drop_duplicates('text_hash').set_index('text_hash')['cleaned_text']
            texts = first.reindex(missing['text_hash']).tolist()

            # Reddit post lengths vary enormously (a one-line title next to a
            # thousand-word essay). Batching them in arbitrary order pads every
            # batch to its longest member, so most of the compute is spent on
            # padding. Sorting by length first makes each batch homogeneous;
            # the original order is restored afterwards.
            order = np.argsort([len(t) for t in texts], kind='stable')
            sorted_texts = [texts[i] for i in order]

            fin_s = np.empty(len(texts))
            rob_s = np.empty(len(texts))
            pos = 0
            for i in tqdm(range(0, len(sorted_texts), self.batch_size),
                          total=math.ceil(len(sorted_texts)/self.batch_size), desc='NLP'):
                chunk = sorted_texts[i:i+self.batch_size]
                f, r = self.score(chunk)
                fin_s[pos:pos+len(chunk)] = f
                rob_s[pos:pos+len(chunk)] = r
                pos += len(chunk)

            fin = np.empty(len(texts))
            rob = np.empty(len(texts))
            fin[order] = fin_s
            rob[order] = rob_s
            missing['finbert'] = fin
            missing['roberta'] = rob

            cache = pd.concat(
                [cache, missing[['text_hash', 'finbert', 'roberta']].astype(
                    {'finbert': 'float32', 'roberta': 'float32'})],
                ignore_index=True).drop_duplicates('text_hash')
            cache.to_parquet(CACHE, index=False)
            print(f'Cache updated -> {len(cache):,} entries')

            todo = todo.set_index('text_hash')
            todo.loc[missing['text_hash'], ['finbert', 'roberta']] = \
                missing[['finbert', 'roberta']].values
            todo = todo.reset_index()

        scores = todo.set_index('text_hash')
        df['finbert'] = scores['finbert'].reindex(df['text_hash']).values
        df['roberta'] = scores['roberta'].reindex(df['text_hash']).values
        df = df.drop(columns=['cleaned_text', 'text_hash'])
        df.to_parquet(OUT, index=False, compression='zstd')
        print(f'Saved {len(df):,} scored posts -> {OUT}')
        print(f'  finbert mean {df["finbert"].mean():+.4f} | '
              f'roberta mean {df["roberta"].mean():+.4f}')


if __name__ == '__main__':
    RedditNLP().run()
