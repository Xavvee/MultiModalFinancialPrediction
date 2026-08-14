import numpy as np
import os
import matplotlib.pyplot as plt
from utils import get_data, calculate_directional_accuracy

def run(ticker, results_dir):
    """Predicts that today's return has the same SIGN as yesterday's.

    This is the directional counterpart of the random walk. A naive last-value
    price forecast implies a change of exactly zero every day, so it makes no
    directional call at all and cannot be scored on directional accuracy - which
    would leave the weak-form-EMH argument without a measurable reference point.
    Carrying yesterday's sign forward keeps the same "no information beyond the
    last observation" premise while producing a genuine call on ~100% of days.

    Beating this is the minimum evidence that a model has found structure in the
    sign sequence rather than merely inheriting its autocorrelation.
    """
    print(f"Persistence Baseline for {ticker}...")
    prices = get_data(ticker)
    returns = prices.pct_change().dropna()

    train_size = int(len(returns) * 0.8)
    test = returns[train_size:]

    # Yesterday's sign, carried forward. The first test day uses the last
    # training day, so no test-set information is used before it is available.
    predictions = np.sign(returns.shift(1)[train_size:].values)

    da = calculate_directional_accuracy(test.values, predictions, is_stationary=True)
    flat = int((predictions == 0).sum())
    print(f"      Directional Accuracy: {da:.2f}%  |  days with no call: {flat}")

    plt.figure(figsize=(14, 5))
    correct = np.sign(test.values) == predictions
    plt.scatter(test.index[correct], test.values[correct], s=14, color='blue',
                label=f'Correct ({correct.sum()})')
    plt.scatter(test.index[~correct], test.values[~correct], s=14, color='red',
                marker='x', label=f'Wrong ({(~correct).sum()})')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f'{ticker}: Persistence Baseline (same direction as yesterday)\n'
              f'Directional Accuracy={da:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"PERSISTENCE_{ticker}.png")
    plt.savefig(save_path)
    plt.close()

    return {'name': 'Persistence (sign)', 'dates': test.index,
            'y_true': test.values, 'y_pred': predictions,
            'is_stationary': True, 'rmse': None}
