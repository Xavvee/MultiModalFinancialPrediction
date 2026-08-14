import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_data(ticker="BTC-USD", start="2017-01-01", end="2019-12-01"):
    print(f"--- Loading data for {ticker} ---")
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data['Close']

def _directions(y_true, y_pred, is_stationary):
    """Derives (true direction, predicted direction) as sign arrays.

    is_stationary=False -> PRICE LEVELS: compare the predicted price against the
                           previous actual price to derive an implied direction
    is_stationary=True  -> RETURNS: compare signs directly
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if is_stationary:
        return np.sign(y_true), np.sign(y_pred)
    return np.sign(y_true[1:] - y_true[:-1]), np.sign(y_pred[1:] - y_true[:-1])


def calculate_directional_accuracy(y_true, y_pred, is_stationary=False):
    """Percentage of correct directional calls, over the days a call was made.

    Predictions implying no change are EXCLUDED from the denominator rather than
    counted as wrong. This matters: a naive persistence forecast predicts exactly
    today's price for tomorrow, so its implied change is zero every single day.
    Scoring those as misses reports 0% accuracy for a model that in truth makes
    no directional call at all - which reads as catastrophic failure instead of
    abstention, and hides the fact that this metric cannot evaluate such a model.

    The exclusion is not free: it makes the denominator model-dependent, so this
    number is only comparable across models when read together with
    directional_coverage(). Report both, or report
    directional_accuracy_strict() alongside.
    """
    true_dir, pred_dir = _directions(y_true, y_pred, is_stationary)
    called = pred_dir != 0
    if called.sum() == 0:
        return float('nan')          # the model never takes a side
    return float((true_dir[called] == pred_dir[called]).mean() * 100)


def directional_accuracy_strict(y_true, y_pred, is_stationary=False):
    """Directional accuracy with no-change predictions counted as WRONG.

    The conservative reading: a model that declines to take a side gets no
    credit for the days it sat out. Keeps the denominator identical for every
    model - so this column is directly comparable across a benchmark table -
    and it penalises the 'lazy predictor' collapse toward a near-constant
    forecast, which the permissive definition above hides.
    """
    true_dir, pred_dir = _directions(y_true, y_pred, is_stationary)
    if len(true_dir) == 0:
        return float('nan')
    return float((true_dir == pred_dir).mean() * 100)


def directional_coverage(y_true, y_pred, is_stationary=False):
    """Share of days on which the model actually took a side.

    Reported alongside accuracy so an abstaining model cannot masquerade as an
    accurate one, or as a failing one.
    """
    _, pred_dir = _directions(y_true, y_pred, is_stationary)
    if len(pred_dir) == 0:
        return float('nan')
    return float((pred_dir != 0).mean() * 100)

def plot_prediction(train, test, prediction, title, filename, metric_name, metric_value):
    dir_acc = calculate_directional_accuracy(test, prediction, is_stationary=False)
    coverage = directional_coverage(test, prediction, is_stationary=False)
    da_text = ("brak sygnalu kierunkowego" if np.isnan(dir_acc)
               else f"Dir. Accuracy={dir_acc:.1f}%"
                    + ("" if coverage > 99.5 else f" (pokrycie {coverage:.0f}%)"))

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train.index, train, label='Training', color='blue')
    plt.plot(test.index, test, label='Real (Test)', color='green')
    plt.plot(test.index, prediction, label='Prediction', color='red', linestyle='--')
    plt.title(f"{title}\n{metric_name}={metric_value:.2f} | {da_text}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    zoom = 50
    plt.plot(test.index[-zoom:], test[-zoom:], label='Real (Test)', color='green', marker='.')
    plt.plot(test.index[-zoom:], prediction[-zoom:], label='Prediction', color='red', linestyle='--', marker='x')
    plt.title(f"ZOOM (Last {zoom} days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Zapisano: {filename} ({da_text})")
    plt.close()