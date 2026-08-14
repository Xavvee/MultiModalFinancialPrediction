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
    """Percentage of correct directional calls.

    is_stationary=False -> PRICE LEVELS: compare the predicted price against the
                           previous actual price to derive an implied direction
    is_stationary=True  -> RETURNS: compare signs directly

    Predictions implying no change are EXCLUDED from the denominator rather than
    counted as wrong. This matters: a naive persistence forecast predicts exactly
    today's price for tomorrow, so its implied change is zero every single day.
    Scoring those as misses reports 0% accuracy for a model that in truth makes
    no directional call at all - which reads as catastrophic failure instead of
    abstention, and hides the fact that this metric cannot evaluate such a model.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if is_stationary:
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
    else:
        true_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])

    called = pred_dir != 0
    if called.sum() == 0:
        return float('nan')          # the model never takes a side
    return float((true_dir[called] == pred_dir[called]).mean() * 100)


def directional_coverage(y_true, y_pred, is_stationary=False):
    """Share of days on which the model actually took a side.

    Reported alongside accuracy so an abstaining model cannot masquerade as an
    accurate one, or as a failing one.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if is_stationary:
        pred_dir = np.sign(y_pred)
    else:
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    return float((pred_dir != 0).mean() * 100)

def plot_prediction(train, test, prediction, title, filename, metric_name, metric_value):
    dir_acc = calculate_directional_accuracy(test, prediction, is_stationary=False)
    coverage = directional_coverage(test, prediction, is_stationary=False)
    da_text = ("brak sygnalu kierunkowego" if np.isnan(dir_acc)
               else f"Dir. Accuracy={dir_acc:.1f}%"
                    + ("" if coverage > 99.5 else f" (pokrycie {coverage:.0f}%)"))

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train.index, train, label='Trening')
    plt.plot(test.index, test, label='Rzeczywiste (Test)', color='green')
    plt.plot(test.index, prediction, label='Predykcja', color='red', linestyle='--')
    plt.title(f"{title}\n{metric_name}={metric_value:.2f} | {da_text}")
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
    print(f"✅ Zapisano: {filename} ({da_text})")
    plt.close()