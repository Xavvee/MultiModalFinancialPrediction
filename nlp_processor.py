import pandas as pd
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import os
import math

class NLPProcessor:
    def __init__(self, tweets_csv, market_csv, output_pkl, sample_size=None, use_weights=True, batch_size=64):
        self.tweets_csv = tweets_csv
        self.market_csv = market_csv
        self.output_pkl = output_pkl
        self.sample_size = sample_size
        self.use_weights = use_weights
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
        """Zwraca dwie listy skalarów dla całej PACZKI tekstów naraz"""
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
            
            # RoBERTa: 0=Negative, 1=Neutral, 2=Positive
            roberta_scores = (r_probs[:, 2] - r_probs[:, 0]).tolist()

        return finbert_scores, roberta_scores

    def process(self):
        print(f"\n--- DUAL-STREAM NLP PIPELINE: {'WEIGHTED' if self.use_weights else 'UNWEIGHTED'} ---")
        df = pd.read_csv(self.tweets_csv)
        
        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42).copy()
        df.reset_index(drop=True, inplace=True)
        
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
        
        if self.use_weights and 'engagement_weight' in df.columns:
            df['finbert_sentiment'] = df['finbert_base'] * df['engagement_weight']
            df['roberta_sentiment'] = df['roberta_base'] * df['engagement_weight']
        else:
            df['finbert_sentiment'] = df['finbert_base']
            df['roberta_sentiment'] = df['roberta_base']
            
        df['combined_sentiment'] = (df['finbert_sentiment'] + df['roberta_sentiment']) / 2.0
            
        print("Aggregating daily sentiment...")
        daily_cols = ['finbert_sentiment', 'roberta_sentiment', 'combined_sentiment']
        daily_sentiment = df.groupby('date')[daily_cols].mean().reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        print(f"Merging with market features from {self.market_csv}...")
        market_df = pd.read_csv(self.market_csv)
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        final_df = pd.merge(market_df, daily_sentiment, on='date', how='inner')
        final_df.sort_values('date', inplace=True)
        
        final_df.to_pickle(self.output_pkl)
        print(f"Success! Final dataset saved to {self.output_pkl}")

if __name__ == "__main__":

    processor_weighted = NLPProcessor(
        'weighted_tweets.csv', 
        'market_features.csv', 
        'full_dataset_weighted.pkl', 
        sample_size=None, 
        use_weights=True,
        batch_size=64 
    )
    processor_weighted.process()