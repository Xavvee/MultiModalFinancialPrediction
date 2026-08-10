import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from utils import calculate_directional_accuracy

# 'sentiment_missing' flags days with zero tweets at all (real gaps in the raw
# 2025-26 collection - only ~201/362 days have any tweets). It travels with the
# market branch since it's a single extra scalar, not worth its own GRU branch.
MARKET_FEATURES = ['daily_return', 'momentum_3d', 'volatility_7d', 'sentiment_missing']
SENTIMENT_STREAMS = ['finbert_whale', 'finbert_retail', 'roberta_whale', 'roberta_retail']
LOOK_BACK = 14
N_FOLDS = 5


def build_windows(market_arr, sentiment_arrs, target_arr, target_indices, look_back):
    """Builds (window, target) pairs for each day index in target_indices,
    using the look_back days strictly preceding it. Indices below look_back
    are skipped (not enough history)."""
    X_market, y = [], []
    X_sent = {s: [] for s in sentiment_arrs}
    for i in target_indices:
        if i < look_back:
            continue
        X_market.append(market_arr[i - look_back:i, :])
        for s in sentiment_arrs:
            X_sent[s].append(sentiment_arrs[s][i - look_back:i, :])
        y.append(target_arr[i])
    X_market = np.array(X_market)
    X_sent = {s: np.array(v) for s, v in X_sent.items()}
    y = np.array(y)
    return X_market, X_sent, y


def build_model(look_back):
    """Multi-input GRU: one small branch per sentiment stream + one market
    branch, fused and reduced to a single up/down probability. Deliberately
    kept small (this dataset has only ~350 usable days) to avoid the model
    just learning to output a safe near-constant prediction."""
    input_market = Input(shape=(look_back, len(MARKET_FEATURES)), name='Input_Market')
    gru_market = GRU(16, name='GRU_Market')(input_market)
    gru_market = Dropout(0.4)(gru_market)

    sentiment_inputs = []
    sentiment_branches = []
    for stream in SENTIMENT_STREAMS:
        inp = Input(shape=(look_back, 1), name=f'Input_{stream}')
        branch = GRU(4, name=f'GRU_{stream}')(inp)
        branch = Dropout(0.4)(branch)
        sentiment_inputs.append(inp)
        sentiment_branches.append(branch)

    fusion_layer = Concatenate(name='Fusion_Concat')([gru_market] + sentiment_branches)
    dense_fusion = Dense(8, activation='relu', name='Dense_Decision')(fusion_layer)
    output = Dense(1, activation='sigmoid', name='Direction_Prediction')(dense_fusion)

    model = Model(inputs=[input_market] + sentiment_inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def run(ticker, results_dir, dataset_pkl='data/new_dataset/processed/full_dataset_whales_2025_26.pkl'):
    # ==========================================
    # EXPERIMENT TRACKING: Change this name before every new idea!
    experiment_name = "V1_Classification_WalkForward"
    # ==========================================

    print(f"GRU Multi-Modal (Whale/Retail) [{experiment_name}] for {ticker}...")

    if not os.path.exists(dataset_pkl):
        print(f"      ERROR: Cannot find {dataset_pkl}. Skipping GRU Multi-modal.")
        return

    df = pd.read_pickle(dataset_pkl)

    if ticker != "BTC-USD":
        print(f"      INFO: NLP dataset is specifically for BTC-USD. Skipping for {ticker}.")
        return

    required_cols = MARKET_FEATURES + SENTIMENT_STREAMS
    df = df.dropna(subset=required_cols).sort_values('date').reset_index(drop=True)

    market_raw = df[MARKET_FEATURES].values
    sentiment_raw = {s: df[s].values.reshape(-1, 1) for s in SENTIMENT_STREAMS}
    target_returns = df['daily_return'].values
    dates = df['date'].values

    valid_targets = np.arange(LOOK_BACK, len(df))
    if len(valid_targets) < N_FOLDS * 2:
        print(f"      ERROR: Not enough days ({len(df)}) for {N_FOLDS}-fold walk-forward with LOOK_BACK={LOOK_BACK}.")
        return

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_da = []
    oof_dates, oof_returns, oof_probs = [], [], []

    for fold, (train_pos, test_pos) in enumerate(tscv.split(valid_targets), start=1):
        train_idx = valid_targets[train_pos]
        test_idx = valid_targets[test_pos]

        # Fit the market-feature scaler on this fold's training days only, then
        # transform the whole series with it so test windows can still borrow
        # look_back context from the (already-scaled) preceding training days.
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
        model.fit(
            train_inputs, y_train_labels,
            epochs=60, batch_size=16, verbose=0,
            validation_split=0.15, callbacks=[early_stop],
        )

        test_probs = model.predict(test_inputs, verbose=0)[:, 0]
        pred_direction = np.where(test_probs >= 0.5, 1.0, -1.0)
        fold_accuracy = calculate_directional_accuracy(y_test_returns, pred_direction, is_stationary=True)
        fold_da.append(fold_accuracy)
        print(f"      Fold {fold}/{N_FOLDS}: train={len(train_idx)} days, test={len(test_idx)} days, DA={fold_accuracy:.2f}%")

        oof_dates.append(dates[test_idx])
        oof_returns.append(y_test_returns)
        oof_probs.append(test_probs)

    oof_dates = np.concatenate(oof_dates)
    oof_returns = np.concatenate(oof_returns)
    oof_probs = np.concatenate(oof_probs)
    oof_direction = np.where(oof_probs >= 0.5, 1.0, -1.0)
    oof_labels = (oof_returns > 0).astype(float)

    overall_da = calculate_directional_accuracy(oof_returns, oof_direction, is_stationary=True)
    overall_loss = log_loss(oof_labels, np.clip(oof_probs, 1e-7, 1 - 1e-7))

    print(f"      Walk-forward summary: per-fold DA = {[f'{d:.1f}%' for d in fold_da]}")
    print(f"      Overall out-of-fold Directional Accuracy: {overall_da:.2f}% | Binary Cross-Entropy: {overall_loss:.4f}")

    order = np.argsort(oof_dates)
    oof_dates, oof_returns, oof_direction = oof_dates[order], oof_returns[order], oof_direction[order]
    correct = np.sign(oof_returns) == oof_direction

    plt.figure(figsize=(14, 7))
    plt.plot(oof_dates, oof_returns, label='Actual Returns', color='green', alpha=0.5)
    plt.scatter(oof_dates[correct], oof_returns[correct], color='blue', marker='o', s=25, label='Correct direction', zorder=3)
    plt.scatter(oof_dates[~correct], oof_returns[~correct], color='red', marker='x', s=25, label='Wrong direction', zorder=3)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    plt.title(f'{ticker}: Multi-Input GRU (Whale/Retail) [{experiment_name}]\n'
              f'Walk-forward OOF Directional Accuracy={overall_da:.2f}% | BCE={overall_loss:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(results_dir, f"GRU_{experiment_name}_{ticker}.png")
    plt.savefig(save_path)
    plt.close()
