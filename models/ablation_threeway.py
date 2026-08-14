"""Three-way GRU ablation: does either sentiment stream, or both, help over price alone?

Same architecture and train/test split as gru_multimodal.py (regression on
daily_return, LOOK_BACK=14, scaler fit on train only), repeated over many random
seeds so the four variants can be compared with a paired test instead of eyeballing
one run each. "Paired" here means literal: every seed sees the identical
train/test split, so seed-to-seed noise (weight init, dropout masks) is the only
thing that differs between a variant's runs - exactly what a paired t-test needs.

Variants:
  price      - daily_return, momentum_3d, volatility_7d
  +finbert   - price + finbert_sentiment
  +roberta   - price + roberta_sentiment
  +both      - price + finbert_sentiment + roberta_sentiment   (= gru_multimodal.py)

Run: python -m models.ablation_threeway [n_seeds]
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization

from utils import calculate_directional_accuracy

DATASET_PKL = 'data/old_dataset/processed/full_dataset_weighted.pkl'
LOOK_BACK = 14
BASE = ['daily_return', 'momentum_3d', 'volatility_7d']
VARIANTS = {
    'price':    BASE,
    '+finbert': BASE + ['finbert_sentiment'],
    '+roberta': BASE + ['roberta_sentiment'],
    '+both':    BASE + ['finbert_sentiment', 'roberta_sentiment'],
}


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


def run_one(seed, variant_cols, train_raw, test_raw, train_target, test_target):
    tf.keras.utils.set_random_seed(seed)

    col_idx = [ALL_COLS.index(c) for c in variant_cols]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_f = scaler.fit_transform(train_raw[:, col_idx])
    test_f = scaler.transform(test_raw[:, col_idx])

    X_train, y_train = create_multivariate_dataset(train_f, train_target, LOOK_BACK)
    X_test, y_test = create_multivariate_dataset(test_f, test_target, LOOK_BACK)

    model = build_model(len(variant_cols))
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    pred = model.predict(X_test, verbose=0)[:, 0]
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    da = calculate_directional_accuracy(y_test, pred, is_stationary=True)
    return rmse, da, y_test


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    df = pd.read_pickle(DATASET_PKL)
    df = df[df['date'] >= '2017-01-01'].copy()
    global ALL_COLS
    ALL_COLS = ['daily_return', 'momentum_3d', 'volatility_7d',
                'finbert_sentiment', 'roberta_sentiment']
    df = df.dropna(subset=ALL_COLS).sort_values('date').reset_index(drop=True)

    all_raw = df[ALL_COLS].values
    target = df['daily_return'].values
    train_size = int(len(df) * 0.8)

    train_raw, test_raw = all_raw[:train_size], all_raw[train_size:]
    train_target, test_target = target[:train_size], target[train_size:]

    # Majority baseline on the EXACT same test period (not a separate yfinance
    # pull, so there's no date-alignment ambiguity with the ablation's own split).
    up_share_train = (train_target > 0).mean()
    maj_direction = 1.0 if up_share_train >= 0.5 else -1.0
    # align to the windowed test target used by the GRU (same offset as create_multivariate_dataset)
    _, y_test_ref = create_multivariate_dataset(
        np.zeros((len(test_raw), 1)), test_target, LOOK_BACK)
    maj_pred = np.full(len(y_test_ref), maj_direction)
    majority_da = calculate_directional_accuracy(y_test_ref, maj_pred, is_stationary=True)
    print(f"Majority baseline on this split: {majority_da:.2f}% "
          f"(always {'UP' if maj_direction > 0 else 'DOWN'}, n_test={len(y_test_ref)})\n")

    records = []
    for seed in range(n_seeds):
        print(f"=== seed {seed+1}/{n_seeds} ===")
        for name, cols in VARIANTS.items():
            rmse, da, _ = run_one(seed, cols, train_raw, test_raw, train_target, test_target)
            print(f"  {name:10s} RMSE={rmse:.4f}  DA={da:.2f}%")
            records.append({'seed': seed, 'variant': name, 'rmse': rmse, 'da': da})

    res = pd.DataFrame(records)
    os.makedirs('results', exist_ok=True)
    res.to_csv('results/ablation_threeway_raw.csv', index=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY over {n_seeds} seeds (majority baseline = {majority_da:.2f}%)")
    print("=" * 70)
    summary = {}
    for name in VARIANTS:
        d = res[res['variant'] == name]['da'].values
        t_vs_maj, p_vs_maj = stats.ttest_1samp(d, majority_da)
        print(f"  {name:10s} DA mean={d.mean():.2f}%  sd={d.std(ddof=1):.2f}  "
              f"min={d.min():.2f}  max={d.max():.2f}  "
              f"| vs majority: t={t_vs_maj:+.2f} p={p_vs_maj:.4f}")
        summary[name] = {'mean_da': float(d.mean()), 'sd_da': float(d.std(ddof=1)),
                          'min_da': float(d.min()), 'max_da': float(d.max()),
                          't_vs_majority': float(t_vs_maj), 'p_vs_majority': float(p_vs_maj)}

    print("\nPairwise paired tests (same seed = same train/test split for both variants):")
    pairs = [('price', '+finbert'), ('price', '+roberta'), ('price', '+both'),
             ('+finbert', '+roberta'), ('+finbert', '+both'), ('+roberta', '+both')]
    pairwise = {}
    for a, b in pairs:
        da_a = res[res['variant'] == a].sort_values('seed')['da'].values
        da_b = res[res['variant'] == b].sort_values('seed')['da'].values
        t, p = stats.ttest_rel(da_a, da_b)
        try:
            w, pw = stats.wilcoxon(da_a, da_b)
        except ValueError:
            w, pw = float('nan'), float('nan')
        diff = da_a.mean() - da_b.mean()
        print(f"  {a:10s} vs {b:10s}  mean diff={diff:+.3f} pp  "
              f"paired-t: t={t:+.2f} p={p:.4f}   wilcoxon: p={pw:.4f}")
        pairwise[f'{a}_vs_{b}'] = {'mean_diff_pp': float(diff), 't': float(t), 'p_ttest': float(p),
                                    'p_wilcoxon': float(pw) if pw == pw else None}

    with open('results/ablation_threeway_summary.json', 'w') as f:
        json.dump({'n_seeds': n_seeds, 'majority_da': float(majority_da),
                    'variants': summary, 'pairwise': pairwise}, f, indent=2)
    print("\nSaved results/ablation_threeway_raw.csv and results/ablation_threeway_summary.json")


if __name__ == '__main__':
    main()
