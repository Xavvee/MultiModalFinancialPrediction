import pandas as pd
import numpy as np
import yfinance as yf
import os

class MarketProcessor:
    def __init__(self, tweets_file='weighted_tweets_2025_26.csv', output_file='market_features_2025_26.csv', ticker='BTC-USD'):
        self.tweets_file = tweets_file
        self.output_file = output_file
        self.ticker = ticker

    def process_market_data(self):
        print(f"--- STARTING MARKET FEATURE ENGINEERING ---")
        
        print(f"Scanning {self.tweets_file} for date ranges...")
        
        df_dates = pd.read_csv(
            self.tweets_file, 
            usecols=['date'], 
            on_bad_lines='skip', 
            engine='python'
        )
        
        # MAGICZNA SZTUCZKA: errors='coerce' zamienia teksty tweetów na NaT, a dropna() je usuwa
        valid_dates = pd.to_datetime(df_dates['date'], errors='coerce').dropna()
        
        start_date = valid_dates.min()
        end_date = valid_dates.max()
        
        print(f"Downloading {self.ticker} data from {start_date.date()} to {end_date.date()}...")

        fetch_start = start_date - pd.Timedelta(days=14) 
        market_df = yf.download(self.ticker, start=fetch_start, end=end_date + pd.Timedelta(days=1))
        
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = market_df.columns.droplevel(1)
            
        market_df.reset_index(inplace=True)
        market_df.rename(columns={'Date': 'date'}, inplace=True)
        market_df['date'] = pd.to_datetime(market_df['date']).dt.tz_localize(None).dt.floor('D')
        
        print("Calculating financial indicators (Returns, Momentum, Volatility)...")
        
        market_df['daily_return'] = market_df['Close'].pct_change()
        
        market_df['momentum_3d'] = market_df['Close'].pct_change(periods=3)
        market_df['momentum_7d'] = market_df['Close'].pct_change(periods=7)
        
        market_df['volatility_7d'] = market_df['daily_return'].rolling(window=7).std()
        
        market_df['volume_change'] = market_df['Volume'].pct_change()

        market_df = market_df.dropna()
        
        market_df = market_df[(market_df['date'] >= start_date) & (market_df['date'] <= end_date)]
        
        final_cols = ['date', 'Close', 'Volume', 'daily_return', 'momentum_3d', 'momentum_7d', 'volatility_7d', 'volume_change']
        final_market_df = market_df[final_cols]
        
        final_market_df.to_csv(self.output_file, index=False)
        print("\n" + "="*50)
        print("--- MARKET ENGINEERING COMPLETE ---")
        print(f"Ready indicators saved to {self.output_file}")
        print("="*50 + "\n")

if __name__ == "__main__":
    processor = MarketProcessor(
        tweets_file='weighted_tweets_2025_26.csv', 
        output_file='market_features_2025_26.csv', 
        ticker='BTC-USD'
    )
    processor.process_market_data()