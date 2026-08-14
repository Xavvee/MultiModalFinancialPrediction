"""How stable is the walk-forward Directional Accuracy across random seeds,
and does the ~20pp fold-to-fold spread track fold volatility?

Reuses the exact fold split, windowing and architecture from gru_multimodal.py
(TimeSeriesSplit is deterministic given the data, so every repeat sees the same
5 folds - only weight init / dropout masks change across repeats). Each fold's
date range and volatility are fixed properties of the data, computed once;
DA is repeated N_REPEATS times per fold so a mean +/- sd can be reported
instead of a single run's number.

Run: python -m models.gru_stability [n_repeats]
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from utils import calculate_directional_accuracy
from models.gru_multimodal import (
    MARKET_FEATURES, SENTIMENT_STREAMS, LOOK_BACK, N_FOLDS,
    build_windows, build_model,
)

DATASET_PKL = 'data/new_dataset/processed/full_dataset_whales_2021_23.pkl'


def main():
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    df = pd.read_pickle(DATASET_PKL)
    required_cols = MARKET_FEATURES + SENTIMENT_STREAMS
    df = df.dropna(subset=required_cols).sort_values('date').reset_index(drop=True)

    market_raw = df[MARKET_FEATURES].values
    sentiment_raw = {s: df[s].values.reshape(-1, 1) for s in SENTIMENT_STREAMS}
    target_returns = df['daily_return'].values
    dates = df['date'].values

    valid_targets = np.arange(LOOK_BACK, len(df))
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    folds = list(tscv.split(valid_targets))

    # Fixed per-fold properties: date range and realised volatility of the test
    # window. These don't depend on the model at all, only on the data and the
    # (deterministic) split.
    fold_info = []
    for fold, (train_pos, test_pos) in enumerate(folds, start=1):
        test_idx = valid_targets[test_pos]
        fold_dates = dates[test_idx]
        fold_returns = target_returns[test_idx]
        fold_info.append({
            'fold': fold,
            'start': str(pd.Timestamp(fold_dates.min()).date()),
            'end': str(pd.Timestamp(fold_dates.max()).date()),
            'n_days': int(len(test_idx)),
            'volatility_std': float(np.std(fold_returns)),
            'up_share': float((fold_returns > 0).mean() * 100),
        })
        print(f"Fold {fold}: {fold_info[-1]['start']} -> {fold_info[-1]['end']} "
              f"(n={fold_info[-1]['n_days']}, vol={fold_info[-1]['volatility_std']:.4f}, "
              f"up-days={fold_info[-1]['up_share']:.1f}%)")

    da_matrix = np.full((n_repeats, N_FOLDS), np.nan)

    for rep in range(n_repeats):
        tf.keras.utils.set_random_seed(rep)
        print(f"\n=== repeat {rep+1}/{n_repeats} (seed={rep}) ===")
        for fold, (train_pos, test_pos) in enumerate(folds, start=1):
            train_idx = valid_targets[train_pos]
            test_idx = valid_targets[test_pos]

            market_scaler = MinMaxScaler(feature_range=(0, 1))
            market_scaler.fit(market_raw[train_idx, :])
            market_scaled = market_scaler.transform(market_raw)

            X_train_market, X_train_sent, y_train_returns = build_windows(
                market_scaled, sentiment_raw, target_returns, train_idx, LOOK_BACK)
            X_test_market, X_test_sent, y_test_returns = build_windows(
                market_scaled, sentiment_raw, target_returns, test_idx, LOOK_BACK)

            y_train_labels = (y_train_returns > 0).astype(float)

            model = build_model(LOOK_BACK)
            train_inputs = [X_train_market] + [X_train_sent[s] for s in SENTIMENT_STREAMS]
            test_inputs = [X_test_market] + [X_test_sent[s] for s in SENTIMENT_STREAMS]

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(train_inputs, y_train_labels, epochs=60, batch_size=16, verbose=0,
                      validation_split=0.15, callbacks=[early_stop])

            test_probs = model.predict(test_inputs, verbose=0)[:, 0]
            pred_direction = np.where(test_probs >= 0.5, 1.0, -1.0)
            da = calculate_directional_accuracy(y_test_returns, pred_direction, is_stationary=True)
            da_matrix[rep, fold - 1] = da
            print(f"  fold {fold}: DA={da:.2f}%")

    print("\n" + "=" * 70)
    print(f"STABILITY SUMMARY over {n_repeats} repeats "
          f"(same {N_FOLDS} folds every time, only init/dropout varies)")
    print("=" * 70)
    for i, info in enumerate(fold_info):
        col = da_matrix[:, i]
        info['da_mean'] = float(np.nanmean(col))
        info['da_sd'] = float(np.nanstd(col, ddof=1))
        print(f"  Fold {info['fold']} [{info['start']} -> {info['end']}]  "
              f"vol={info['volatility_std']:.4f}  "
              f"DA mean={info['da_mean']:.2f}%  sd={info['da_sd']:.2f}  "
              f"(single-run range seen: {col.min():.1f}-{col.max():.1f})")

    vols = [f['volatility_std'] for f in fold_info]
    means = [f['da_mean'] for f in fold_info]
    corr = float(np.corrcoef(vols, means)[0, 1]) if len(vols) > 2 else float('nan')
    print(f"\n  Correlation across folds: volatility vs mean DA  r={corr:+.3f}  (n={len(vols)} folds)")

    overall_mean = float(np.nanmean(da_matrix))
    overall_sd_across_reps = float(np.nanstd(np.nanmean(da_matrix, axis=1), ddof=1))
    print(f"\n  Grand mean DA across all repeats x folds: {overall_mean:.2f}%")
    print(f"  SD of the per-repeat overall DA (repeat-to-repeat noise on the aggregate): {overall_sd_across_reps:.2f} pp")

    with open('results/gru_stability_summary.json', 'w') as f:
        json.dump({'n_repeats': n_repeats, 'folds': fold_info,
                    'volatility_vs_da_corr': corr,
                    'grand_mean_da': overall_mean,
                    'overall_da_repeat_sd': overall_sd_across_reps}, f, indent=2)
    np.save('results/gru_stability_da_matrix.npy', da_matrix)
    print("\nSaved results/gru_stability_summary.json and results/gru_stability_da_matrix.npy")


if __name__ == '__main__':
    main()
