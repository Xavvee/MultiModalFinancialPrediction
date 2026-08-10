import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_data(ticker="BTC-USD", start="2020-01-01", end="2026-03-03"):
    print(f"--- Loading data for {ticker} ---")
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data['Close']

def calculate_directional_accuracy(y_true, y_pred, is_stationary=False):
    """
    Calculates the percentage of correct directional predictions.
    is_stationary=False -> for PRICE LEVELS (we check if the predicted price is above or below the previous actual price)
    is_stationary=True  -> for RETURNS (we calculate the sign of the differences)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct_directions = 0
    total_steps = len(y_true)
    
    if is_stationary:
        for i in range(total_steps):
            if np.sign(y_true[i]) == np.sign(y_pred[i]):
                correct_directions += 1
    else:
        total_steps = len(y_true) - 1
        for i in range(1, len(y_true)):
            true_change = y_true[i] - y_true[i-1]
            pred_change = y_pred[i] - y_true[i-1]
            if np.sign(true_change) == np.sign(pred_change):
                correct_directions += 1
            
    return (correct_directions / total_steps) * 100

def plot_prediction(train, test, prediction, title, filename, metric_name, metric_value):
    dir_acc = calculate_directional_accuracy(test, prediction, is_stationary=False)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(train.index, train, label='Trening')
    plt.plot(test.index, test, label='Rzeczywiste (Test)', color='green')
    plt.plot(test.index, prediction, label='Predykcja', color='red', linestyle='--')
    plt.title(f"{title}\n{metric_name}={metric_value:.2f} | Dir. Accuracy={dir_acc:.1f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    zoom = 50
    plt.plot(test.index[-zoom:], test[-zoom:], label='Rzeczywiste', color='green', marker='.')
    plt.plot(test.index[-zoom:], prediction[-zoom:], label='Predykcja', color='red', linestyle='--', marker='x')
    plt.title(f"ZOOM (Ostatnie {zoom} dni)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Zapisano: {filename} (DA: {dir_acc:.1f}%)")
    plt.close()