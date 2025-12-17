import os
import time
import shutil
from models import random_walk, arima, arima_stationary, lstm, lstm_stationary, dashboard

RESULTS_DIR = "results"
ASSETS = ["BTC-USD", "ETH-USD", "^GSPC"]

def clean_results_directory(directory):
    """Usuwa całą zawartość katalogu results, żeby mieć czysto przed startem."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def main():
    print(f"Wyniki trafią do: ./{RESULTS_DIR}/")
    
    clean_results_directory(RESULTS_DIR)
    start_time = time.time()

    for ticker in ASSETS:
        print(f"\n" + "="*50)
        print(f"PRZETWARZANIE AKTYWA: {ticker}")
        print("="*50)
        
        try:
            asset_dir = os.path.join(RESULTS_DIR, ticker)
            if not os.path.exists(asset_dir):
                os.makedirs(asset_dir)

            # 1. Random Walk
            random_walk.run(ticker, asset_dir)
            
            # 2. ARIMA
            arima.run(ticker, asset_dir)

            # 3. ARIMA (Zwroty - NOWOŚĆ)
            arima_stationary.run(ticker, asset_dir)

            # 4. LSTM (Ceny)
            lstm.run(ticker, asset_dir)
            
            # 5. LSTM (Zwroty)
            lstm_stationary.run(ticker, asset_dir)

            # 6. DASHBOARD
            dashboard.run(ticker, asset_dir)
            
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD dla {ticker}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nZakończono wszystko. Czas: {elapsed:.2f} s")

if __name__ == "__main__":
    main()