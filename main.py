import os
import time
import shutil
import sentiment_ab_test
from models import (random_walk, majority_baseline, persistence_baseline,
                    arima, arima_stationary, lstm, lstm_stationary,
                    gru_multimodal, dashboard, benchmark_table)

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

    print("\n" + "="*50)
    print("Generating Sentiment A/B Test Plot...")
    print("="*50)
    try:
        sentiment_ab_test.plot_sentiment_ab_test(
            weighted_pkl='data/old_dataset/processed/full_dataset_weighted.pkl',
            unweighted_pkl='data/old_dataset/archive/dataset_unweighted.pkl',
            output_file=os.path.join(RESULTS_DIR, 'sentiment_ab_test.png')
        )
    except Exception as e:
        print(f"Failed to generate Sentiment A/B Test: {e}")
        print("Make sure both weighted and unweighted .pkl files exist!")

    for ticker in ASSETS:
        print(f"\n" + "="*50)
        print(f"Processed assets: {ticker}")
        print("="*50)
        
        try:
            asset_dir = os.path.join(RESULTS_DIR, ticker)
            if not os.path.exists(asset_dir):
                os.makedirs(asset_dir)

            # Every run() returns its dated predictions so the benchmark table
            # can score them all on one common window (see models/benchmark_table.py).
            collected = []

            # 1. Random Walk
            collected.append(random_walk.run(ticker, asset_dir))

            # 1b. Majority baseline - the directional bar every model must clear
            collected.append(majority_baseline.run(ticker, asset_dir))

            # 1c. Persistence baseline - the random walk's measurable counterpart
            collected.append(persistence_baseline.run(ticker, asset_dir))

            # 2. ARIMA
            collected.append(arima.run(ticker, asset_dir))

            # 3. ARIMA (Returns)
            collected.append(arima_stationary.run(ticker, asset_dir))

            # 4. LSTM (Prices)
            collected.append(lstm.run(ticker, asset_dir))

            # 5. LSTM (Returns)
            collected.append(lstm_stationary.run(ticker, asset_dir))

            # 6. Multi-Modal GRU (Returns + Dual-Stream Sentiment)
            collected.append(gru_multimodal.run(
                ticker, asset_dir,
                dataset_pkl='data/old_dataset/processed/full_dataset_weighted.pkl'))

            # 7. DASHBOARD
            dashboard.run(ticker, asset_dir)

            # 8. One comparable table across all of the above
            benchmark_table.build(collected, ticker, asset_dir)
            
        except Exception as e:
            print(f"CRITICAL ERROR for {ticker}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nFinished! Total benchmark execution time: {elapsed:.2f} s")

if __name__ == "__main__":
    main()