"""Direct test for the 'lazy predictor' collapse (Pintelas et al.).

Directional accuracy near the majority baseline is only INDIRECT evidence. The
direct signature is variance: a network trained under MSE on a near-random-walk
target minimises loss most cheaply by predicting something close to the constant
mean, so its forecasts have far less spread than the series it is forecasting.

Two statistics, neither of which depends on directional accuracy:

  variance ratio  sd(y_pred) / sd(y_true)
      1.0 means the forecast is as volatile as reality; approaching 0 means the
      model has given up on amplitude and is emitting a near-constant.

  central mass    share of predictions inside the middle decile of the actual
      return distribution
      A constant forecast parks nearly everything there; a genuinely varying
      forecast spreads out roughly like the target does (~10%).

This also explains the coverage column in the benchmark: a forecast collapsed
onto a constant that happens to sit near zero produces almost no directional
calls at all.

Run: python -m models.lazy_predictor_test
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization

from utils import calculate_directional_accuracy

DATASET_PKL = 'data/old_dataset/processed/full_dataset_weighted.pkl'
LOOK_BACK = 14
BASE = ['daily_return', 'momentum_3d', 'volatility_7d']
SENT = ['finbert_sentiment', 'roberta_sentiment']
N_SEEDS = 5


def create_multivariate_dataset(features, target, look_back):
    X, y = [], []
    for i in range(len(features) - look_back - 1):
        X.append(features[i:(i + look_back), :])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)


def build_model(n_features):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=(LOOK_BACK, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(GRU(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    df = pd.read_pickle(DATASET_PKL)
    df = df[df['date'] >= '2017-01-01'].copy()
    cols = BASE + SENT
    df = df.dropna(subset=cols).sort_values('date').reset_index(drop=True)

    raw = df[cols].values
    target = df['daily_return'].values
    train_size = int(len(df) * 0.8)

    variants = {'price only': BASE, 'price + sentiment': BASE + SENT}
    print(f"{'variant':20s} {'seed':>5} {'sd(pred)':>10} {'sd(true)':>10} "
          f"{'ratio':>8} {'central':>9} {'DA':>7}")

    summary = {}
    for label, vcols in variants.items():
        idx = [cols.index(c) for c in vcols]
        ratios, centrals = [], []
        for seed in range(N_SEEDS):
            tf.keras.utils.set_random_seed(seed)
            scaler = MinMaxScaler(feature_range=(0, 1))
            tr = scaler.fit_transform(raw[:train_size, :][:, idx])
            te = scaler.transform(raw[train_size:, :][:, idx])

            X_tr, y_tr = create_multivariate_dataset(tr, target[:train_size], LOOK_BACK)
            X_te, y_te = create_multivariate_dataset(te, target[train_size:], LOOK_BACK)

            model = build_model(len(vcols))
            model.fit(X_tr, y_tr, epochs=25, batch_size=32, verbose=0)
            pred = model.predict(X_te, verbose=0)[:, 0]

            sd_p, sd_t = float(np.std(pred)), float(np.std(y_te))
            ratio = sd_p / sd_t
            lo, hi = np.percentile(y_te, [45, 55])
            central = float(((pred >= lo) & (pred <= hi)).mean() * 100)
            da = calculate_directional_accuracy(y_te, pred, is_stationary=True)

            ratios.append(ratio)
            centrals.append(central)
            print(f"{label:20s} {seed:>5} {sd_p:10.5f} {sd_t:10.5f} "
                  f"{ratio:8.3f} {central:8.1f}% {da:6.2f}%")

        summary[label] = (float(np.mean(ratios)), float(np.mean(centrals)))

    print("\n" + "=" * 70)
    print("LAZY PREDICTOR VERDICT")
    print("=" * 70)
    for label, (r, c) in summary.items():
        print(f"  {label:20s} variance ratio {r:.3f}   central mass {c:.1f}% "
              f"(a varying forecast would sit near 10%)")
    worst = min(summary.values())[0]
    print()
    if worst < 0.35:
        print("  CONFIRMED: forecast spread is a small fraction of the target's.")
        print("  The network is emitting a near-constant - the lazy-predictor collapse")
        print("  described in Chapter 2, now measured directly rather than inferred")
        print("  from directional accuracy.")
    else:
        print("  NOT CONFIRMED: the forecasts retain substantial spread, so the")
        print("  near-chance directional accuracy is not explained by a collapse")
        print("  toward a constant. The signal is simply absent.")


if __name__ == '__main__':
    main()
