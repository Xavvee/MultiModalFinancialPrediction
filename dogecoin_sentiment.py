import os
import math
import hashlib
import pandas as pd
import numpy as np
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

"""Scores the Musk tweets with the same two models used everywhere else.

The point of the positive control is to test THIS pipeline, so the text goes
through the identical cleaning and the identical models - no special handling
that could make the control easier to pass than the real analyses were.
"""

CACHE = 'data/sentiment_cache.parquet'
IN = 'data/dogecoin/raw/musk_tweets_2021.csv'
OUT = 'data/dogecoin/processed/musk_tweets_scored.parquet'
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def hash_text(s):
    return hashlib.blake2b(str(s).encode('utf-8', 'ignore'), digest_size=8).hexdigest()


def main():
    tw = pd.read_csv(IN)
    tw['ts'] = pd.to_datetime(tw['Datetime'], format='%d/%m/%Y %H:%M', errors='coerce')
    tw = tw[tw['ts'].notna()].reset_index(drop=True)
    tw['cleaned_text'] = tw['Text'].astype(str).map(clean_text)
    print(f'Tweets: {len(tw):,}')

    if torch_directml.is_available():
        device = torch_directml.device()
        print(f'GPU: {device}')
    else:
        device = torch.device('cpu')
        print('CPU fallback')

    f_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    f_mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
    r_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    r_tok = AutoTokenizer.from_pretrained(r_path)
    r_mod = AutoModelForSequenceClassification.from_pretrained(r_path).to(device).eval()

    def score(texts):
        short = [" ".join(str(t).split()[:400]) for t in texts]
        with torch.no_grad():
            fi = f_tok(short, return_tensors='pt', padding=True, truncation=True, max_length=512)
            fi = {k: v.to(device) for k, v in fi.items()}
            fp = torch.nn.functional.softmax(f_mod(**fi).logits, dim=-1).cpu().numpy()
            ri = r_tok(short, return_tensors='pt', padding=True, truncation=True, max_length=512)
            ri = {k: v.to(device) for k, v in ri.items()}
            rp = torch.nn.functional.softmax(r_mod(**ri).logits, dim=-1).cpu().numpy()
        return (fp[:, 0] - fp[:, 1]).tolist(), (rp[:, 2] - rp[:, 0]).tolist()

    fin, rob = [], []
    bs = 64
    for i in tqdm(range(0, len(tw), bs), total=math.ceil(len(tw)/bs), desc='NLP'):
        f, r = score(tw['cleaned_text'].iloc[i:i+bs].tolist())
        fin.extend(f)
        rob.extend(r)
    tw['finbert'] = fin
    tw['roberta'] = rob

    # fold into the shared cache so nothing is ever recomputed
    tw['text_hash'] = tw['cleaned_text'].map(hash_text)
    if os.path.exists(CACHE):
        cache = pd.read_parquet(CACHE)
    else:
        cache = pd.DataFrame(columns=['text_hash', 'finbert', 'roberta'])
    add = tw[['text_hash', 'finbert', 'roberta']].astype(
        {'finbert': 'float32', 'roberta': 'float32'})
    cache = pd.concat([cache, add], ignore_index=True).drop_duplicates('text_hash')
    cache.to_parquet(CACHE, index=False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tw[['ts', 'finbert', 'roberta']].to_parquet(OUT, index=False)
    print(f'Saved -> {OUT}   cache now {len(cache):,} entries')
    print(f'  finbert mean {tw["finbert"].mean():+.4f} | roberta mean {tw["roberta"].mean():+.4f}')


if __name__ == '__main__':
    main()
