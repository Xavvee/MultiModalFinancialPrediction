import os
import time
import shutil
from models import random_walk, arima, arima_stationary, lstm, lstm_stationary, gru_multimodal, dashboard

RESULTS_DIR = "results"
ASSETS = ["BTC-USD", "ETH-USD", "^GSPC"]

def clean_results_directory(directory):
    """Deleting all contents of the results directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def main():
    print(f"Results will be saved to: ./{RESULTS_DIR}/")
    
    clean_results_directory(RESULTS_DIR)
    start_time = time.time()

    for ticker in ASSETS:
        print(f"\n" + "="*50)
        print(f"Processed assets: {ticker}")
        print("="*50)
        
        try:
            asset_dir = os.path.join(RESULTS_DIR, ticker)
            if not os.path.exists(asset_dir):
                os.makedirs(asset_dir)

            # 1. Random Walk
            random_walk.run(ticker, asset_dir)
            
            # 2. ARIMA
            arima.run(ticker, asset_dir)

            # 3. ARIMA (Returns)
            arima_stationary.run(ticker, asset_dir)

            # 4. LSTM (Prices)
            lstm.run(ticker, asset_dir)
            
            # 5. LSTM (Returns)
            lstm_stationary.run(ticker, asset_dir)

            # 6. Multi-Modal GRU (Returns + Whale/Retail Multi-Input Sentiment)
            # This model automatically skips non-BTC assets internally
            gru_multimodal.run(ticker, asset_dir, dataset_pkl='data/new_dataset/processed/full_dataset_whales_2025_26.pkl')

            # 7. DASHBOARD
            dashboard.run(ticker, asset_dir)
            
        except Exception as e:
            print(f"CRITICAL ERROR for {ticker}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nFinished! Total benchmark execution time: {elapsed:.2f} s")

if __name__ == "__main__":
    main()