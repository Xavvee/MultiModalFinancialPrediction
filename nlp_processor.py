import pandas as pd
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import os
import math

class NLPProcessor:
    def __init__(self, tweets_csv, market_csv, output_pkl, sample_size=None, batch_size=64):
        self.tweets_csv = tweets_csv
        self.market_csv = market_csv
        self.output_pkl = output_pkl
        self.sample_size = sample_size
        self.batch_size = batch_size
        
        if torch_directml.is_available():
            self.device = torch_directml.device()
            print(f"Loading Dual-NLP models on AMD Radeon: {self.device}...")
        else:
            self.device = torch.device('cpu')
            print("Warning: DirectML not found. Falling back to CPU...")
            
        print("Loading FinBERT (Wall Street logic)...")
        self.finbert_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.finbert_mod.eval()

        print("Loading Twitter-RoBERTa (Social Media logic)...")
        roberta_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
        self.roberta_mod = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(self.device)
        self.roberta_mod.eval()

    def get_dual_sentiment_batch(self, texts):
        short_texts = [" ".join(str(t).split()[:400]) if isinstance(t, str) else "" for t in texts]

        with torch.no_grad():
            # --- FINBERT BATCH INFERENCE ---
            f_inputs = self.finbert_tok(short_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            f_inputs = {k: v.to(self.device) for k, v in f_inputs.items()}
            f_outputs = self.finbert_mod(**f_inputs)
            f_probs = torch.nn.functional.softmax(f_outputs.logits, dim=-1).cpu().numpy()
            finbert_scores = (f_probs[:, 0] - f_probs[:, 1]).tolist()

            # --- ROBERTA BATCH INFERENCE ---
            r_inputs = self.roberta_tok(short_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            r_inputs = {k: v.to(self.device) for k, v in r_inputs.items()}
            r_outputs = self.roberta_mod(**r_inputs)
            r_probs = torch.nn.functional.softmax(r_outputs.logits, dim=-1).cpu().numpy()
            roberta_scores = (r_probs[:, 2] - r_probs[:, 0]).tolist()

        return finbert_scores, roberta_scores

    def process(self):
        print(f"\n--- DUAL-STREAM NLP PIPELINE (WHALE vs RETAIL) ---")
        df = pd.read_csv(self.tweets_csv, engine='python', on_bad_lines='skip')

        # Floor to calendar day. Existing CSVs generated before this fix still
        # carry full "YYYY-MM-DD HH:MM:SS" timestamps, which almost never match
        # the midnight-floored dates in market_features_*.csv once grouped below.
        df['date'] = df['date'].astype(str).str.slice(0, 10)

        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42).copy()
        df.reset_index(drop=True, inplace=True)
        
        output_dir = os.path.dirname(self.output_pkl)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        checkpoint_file = self.output_pkl.replace('.pkl', '_checkpoint.csv')

        if os.path.exists(checkpoint_file):
            checkpoint_df = pd.read_csv(checkpoint_file)
            start_idx = len(checkpoint_df)
            f_scores = checkpoint_df['finbert_base'].tolist()
            r_scores = checkpoint_df['roberta_base'].tolist()
            print(f"Resuming from checkpoint! Already processed: {start_idx} / {len(df)} tweets.")
        else:
            f_scores = []
            r_scores = []
            start_idx = 0
            print(f"Starting fresh. Total to process: {len(df)} tweets.")
            
        save_interval = 100_000
        total_batches = math.ceil((len(df) - start_idx) / self.batch_size)
        
        # --- BATCH INFERENCE LOOP ---
        for i in tqdm(range(start_idx, len(df), self.batch_size), total=total_batches, desc="Dual NLP Batch Inference"):
            batch_texts = df['cleaned_text'].iloc[i:i+self.batch_size].tolist()
            
            f_batch, r_batch = self.get_dual_sentiment_batch(batch_texts)
            f_scores.extend(f_batch)
            r_scores.extend(r_batch)
            
            current_processed = len(f_scores)
            if current_processed % save_interval < self.batch_size and current_processed > start_idx:
                pd.DataFrame({'finbert_base': f_scores, 'roberta_base': r_scores}).to_csv(checkpoint_file, index=False)
                tqdm.write(f"[Checkpoint] Safety save at {current_processed} tweets.")
        
        pd.DataFrame({'finbert_base': f_scores, 'roberta_base': r_scores}).to_csv(checkpoint_file, index=False)
            
        df['finbert_base'] = f_scores
        df['roberta_base'] = r_scores
        
        # --- NOWA LOGIKA: ROZDZIELENIE KOHORT ---
        print("Aggregating daily sentiment (Splitting Whales and Retail)...")

        # A handful of rows have a corrupted 'date' field (row-misalignment
        # artifact from a legacy CSV-escaping bug upstream in the raw data),
        # e.g. literal tweet text instead of a date. Drop those before grouping
        # so they can't crash the date parsing or land in the wrong day's bucket.
        valid_date_mask = pd.to_datetime(df['date'], errors='coerce').notna()
        dropped = (~valid_date_mask).sum()
        if dropped:
            print(f"      WARNING: Dropping {dropped} rows with an unparseable 'date' field.")
        df = df[valid_date_mask]

        # Flaga wieloryba: waga > 1.0 (zgodnie z naszym feature_engineer.py)
        df['is_whale'] = df['engagement_weight'] > 1.0

        # Grupowanie po dacie oraz fladze wieloryba
        daily_grouped = df.groupby(['date', 'is_whale'])[['finbert_base', 'roberta_base']].mean().unstack()
        
        # Spłaszczanie nazw kolumn z MultiIndexu
        new_cols = []
        for col in daily_grouped.columns:
            model_name = col[0].replace('_base', '') # finbert lub roberta
            suffix = "whale" if col[1] == True else "retail"
            new_cols.append(f"{model_name}_{suffix}")
            
        daily_grouped.columns = new_cols
        daily_grouped.reset_index(inplace=True)
        daily_grouped['date'] = pd.to_datetime(daily_grouped['date'])
        
        # Jeśli w dany dzień nie było tweetów od wielorybów (ale były retail), wstawiamy 0 (sentyment neutralny)
        daily_grouped = daily_grouped.fillna(0)

        print(f"Merging with market features from {self.market_csv}...")
        market_df = pd.read_csv(self.market_csv)
        market_df['date'] = pd.to_datetime(market_df['date'])

        # LEFT join on the full market calendar (BTC trades daily, no gaps) instead
        # of an inner join, so days with zero tweets stay in the dataset rather than
        # being silently dropped. The raw 2025-26 tweet collection has real calendar
        # gaps of its own (only ~201/362 days have any tweets at all) - dropping
        # those days broke the GRU's look-back windows, which walk across ROW
        # POSITION, not calendar time, and could silently span up to 25 real days
        # within a single window when gap days were missing instead of present.
        final_df = pd.merge(market_df, daily_grouped, on='date', how='left')
        final_df.sort_values('date', inplace=True)

        sentiment_cols = new_cols
        final_df['sentiment_missing'] = final_df[sentiment_cols].isna().all(axis=1).astype(float)
        # Forward-fill sentiment through gap days (yesterday's known mood is a
        # better guess than a false "neutral"); fillna(0) covers any leading gap
        # before the first day that has tweet data at all.
        final_df[sentiment_cols] = final_df[sentiment_cols].ffill().fillna(0)

        final_df.to_pickle(self.output_pkl)
        print(f"Success! Final dataset saved to {self.output_pkl}")
        print(f"Created Sentiment Columns: {new_cols}")
        print(f"Days with no tweets at all (forward-filled, flagged via 'sentiment_missing'): "
              f"{int(final_df['sentiment_missing'].sum())} / {len(final_df)}")

if __name__ == "__main__":
    processor_whale = NLPProcessor(
        tweets_csv='data/new_dataset/interim/weighted_tweets_2025_26.csv',
        market_csv='data/new_dataset/market/market_features_2025_26.csv',
        output_pkl='data/new_dataset/processed/full_dataset_whales_2025_26.pkl',
        sample_size=None,
        batch_size=64
    )
    processor_whale.process()