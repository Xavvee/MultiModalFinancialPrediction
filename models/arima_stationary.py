import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from utils import get_data, calculate_directional_accuracy

def run(ticker, results_dir):
    print(f"ARIMA (Returns/Stationary) dla {ticker}...")
    prices = get_data(ticker)
    returns = prices.pct_change().dropna()
    
    train_size = int(len(returns) * 0.8)
    train, test = returns[0:train_size], returns[train_size:len(returns)]

    history = [x for x in train]
    predictions = []
    
    total = len(test)
    checkpoints = [int(total*0.25), int(total*0.5), int(total*0.75)]

    for t in range(total):
        model = ARIMA(history, order=(5, 0, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
        
        if t in checkpoints:
            print(f"      Postęp: {int(t/total*100)}%")

    predictions_series = pd.Series(predictions, index=test.index)
    rmse = np.sqrt(mean_squared_error(test, predictions_series))
    
    da = calculate_directional_accuracy(test.values, predictions_series.values, is_stationary=True)
    print(f"      Wynik RMSE: {rmse:.6f} | Dir Acc: {da:.2f}%")

    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test, label='Rzeczywiste Zmiany', color='green', alpha=0.5)
    plt.plot(test.index, predictions_series, label='Predykcja ARIMA', color='red', alpha=0.8, linewidth=1.5)
    
    plt.title(f'{ticker}: ARIMA Stationary Forecast\nRMSE={rmse:.6f} | Directional Accuracy={da:.2f}%')
    plt.xlabel('Data')
    plt.ylabel('Zmiana ceny (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(results_dir, f"ARIMA_Returns_{ticker}.png")
    plt.savefig(save_path)
    plt.close()