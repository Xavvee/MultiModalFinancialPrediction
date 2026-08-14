import numpy as np
import os
import matplotlib.pyplot as plt
from utils import get_data, calculate_directional_accuracy

def run(ticker, results_dir):
    """Predicts the majority direction of the training period, every single day.

    This is the baseline any directional model must beat to be worth anything.
    An asset that trended upward over the sample has more up-days than down-days,
    so a constant "up" call already scores above 50% - and models reporting
    55% directional accuracy can quietly be losing to it.

    It is deliberately kept separate from Random Walk: with the price-level
    definition of directional accuracy, a naive last-value forecast predicts a
    change of exactly zero every day, whose sign never matches, so Random Walk
    scores 0% by construction and cannot serve as this reference point.
    """
    print(f"Majority Baseline for {ticker}...")
    prices = get_data(ticker)
    returns = prices.pct_change().dropna()

    train_size = int(len(returns) * 0.8)
    train, test = returns[0:train_size], returns[train_size:len(returns)]

    # The call is fixed from the training period only - no peeking at the test set.
    up_share_train = (train > 0).mean()
    direction = 1.0 if up_share_train >= 0.5 else -1.0
    label = "UP" if direction > 0 else "DOWN"

    predictions = np.full(len(test), direction)
    da = calculate_directional_accuracy(test.values, predictions, is_stationary=True)

    up_share_test = (test > 0).mean() * 100
    print(f"      Always predicting {label} (train up-share {up_share_train*100:.2f}%)")
    print(f"      Directional Accuracy: {da:.2f}%  |  test up-days: {up_share_test:.2f}%")

    plt.figure(figsize=(14, 5))
    correct = np.sign(test.values) == direction
    plt.scatter(test.index[correct], test.values[correct], s=14, color='blue',
                label=f'Correct ({correct.sum()})')
    plt.scatter(test.index[~correct], test.values[~correct], s=14, color='red',
                marker='x', label=f'Wrong ({(~correct).sum()})')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f'{ticker}: Majority Baseline (always {label})\n'
              f'Directional Accuracy={da:.2f}% — the bar every model must clear')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"MAJORITY_{ticker}.png")
    plt.savefig(save_path)
    plt.close()

    return da
