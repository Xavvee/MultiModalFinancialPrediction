import numpy as np
import os
from sklearn.metrics import mean_squared_error
from utils import get_data, plot_prediction

def run(ticker, results_dir):
    print(f"Random Walk dla {ticker}...")
    prices = get_data(ticker)
    train_size = int(len(prices) * 0.8)
    train, test = prices[0:train_size], prices[train_size:len(prices)]

    predictions = test.shift(1)
    predictions.iloc[0] = train.iloc[-1]

    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    save_path = os.path.join(results_dir, f"RW_{ticker}.png")
    plot_prediction(train, test, predictions, f"{ticker}: Random Walk", save_path, "RMSE", rmse)