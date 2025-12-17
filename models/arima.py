import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from utils import get_data, plot_prediction

def run(ticker, results_dir):
    print(f"ARIMA (Rolling) dla {ticker}...")
    prices = get_data(ticker)
    train_size = int(len(prices) * 0.8)
    train, test = prices[0:train_size], prices[train_size:len(prices)]

    history = [x for x in train]
    predictions = []
    
    total = len(test)
    checkpoints = [int(total*0.25), int(total*0.5), int(total*0.75)]

    for t in range(total):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
        
        if t in checkpoints:
            print(f"      Postęp ARIMA: {int(t/total*100)}%")

    predictions_series = pd.Series(predictions, index=test.index)
    rmse = np.sqrt(mean_squared_error(test, predictions_series))

    save_path = os.path.join(results_dir, f"ARIMA_{ticker}.png")
    plot_prediction(train, test, predictions_series, f"{ticker}: ARIMA Rolling", save_path, "RMSE", rmse)