import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate
from utils import calculate_directional_accuracy

MARKET_FEATURES = ['daily_return', 'momentum_3d', 'volatility_7d']
SENTIMENT_STREAMS = ['finbert_whale', 'finbert_retail', 'roberta_whale', 'roberta_retail']
LOOK_BACK = 14


def create_branch_windows(feature_array, look_back):
    """Windows a (days, n_features) array into (samples, look_back, n_features)."""
    X = []
    for i in range(len(feature_array) - look_back - 1):
        X.append(feature_array[i:(i + look_back), :])
    return np.array(X)


def create_target_windows(target_array, look_back):
    Y = []
    for i in range(len(target_array) - look_back - 1):
        Y.append(target_array[i + look_back])
    return np.array(Y)


def run(ticker, results_dir, dataset_pkl='data/new_dataset/processed/full_dataset_whales_2025_26.pkl'):
    # ==========================================
    # EXPERIMENT TRACKING: Change this name before every new idea!
    experiment_name = "V0_MultiInput_WhaleRetail"
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
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    target_returns = df['daily_return'].values
    market_raw = df[MARKET_FEATURES].values

    # Split into train and test sets (80/20 ratio) BEFORE scaling
    train_size = int(len(df) * 0.8)

    market_train_raw = market_raw[0:train_size, :]
    market_test_raw = market_raw[train_size:, :]

    target_train = target_returns[0:train_size]
    target_test = target_returns[train_size:]

    # Fit the market-feature scaler on the training split only, then transform
    # both splits, so test-set statistics never leak into training.
    market_scaler = MinMaxScaler(feature_range=(0, 1))
    market_train = market_scaler.fit_transform(market_train_raw)
    market_test = market_scaler.transform(market_test_raw)

    # Sentiment streams are already bounded in [-1, 1] (softmax-probability
    # differences), so they are used as-is without extra scaling.
    sentiment_train = {s: df[s].values[0:train_size].reshape(-1, 1) for s in SENTIMENT_STREAMS}
    sentiment_test = {s: df[s].values[train_size:].reshape(-1, 1) for s in SENTIMENT_STREAMS}

    X_train_market = create_branch_windows(market_train, LOOK_BACK)
    X_test_market = create_branch_windows(market_test, LOOK_BACK)

    X_train_sent = {s: create_branch_windows(sentiment_train[s], LOOK_BACK) for s in SENTIMENT_STREAMS}
    X_test_sent = {s: create_branch_windows(sentiment_test[s], LOOK_BACK) for s in SENTIMENT_STREAMS}

    y_train = create_target_windows(target_train, LOOK_BACK)
    y_test = create_target_windows(target_test, LOOK_BACK)

    # ---------------------------------------------------------
    # MODEL ARCHITECTURE: MULTI-INPUT (one GRU branch per stream)
    # ---------------------------------------------------------
    input_market = Input(shape=(LOOK_BACK, len(MARKET_FEATURES)), name='Input_Market')
    gru_market = GRU(32, name='GRU_Market')(input_market)
    gru_market = Dropout(0.3)(gru_market)

    sentiment_inputs = []
    sentiment_branches = []
    for stream in SENTIMENT_STREAMS:
        inp = Input(shape=(LOOK_BACK, 1), name=f'Input_{stream}')
        branch = GRU(8, name=f'GRU_{stream}')(inp)
        branch = Dropout(0.3)(branch)
        sentiment_inputs.append(inp)
        sentiment_branches.append(branch)

    fusion_layer = Concatenate(name='Fusion_Concat')([gru_market] + sentiment_branches)
    dense_fusion = Dense(16, activation='relu', name='Dense_Decision')(fusion_layer)
    output = Dense(1, name='Final_Prediction')(dense_fusion)

    model = Model(inputs=[input_market] + sentiment_inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print("      Training multi-input GRU model for Regression...")
    train_inputs = [X_train_market] + [X_train_sent[s] for s in SENTIMENT_STREAMS]
    test_inputs = [X_test_market] + [X_test_sent[s] for s in SENTIMENT_STREAMS]

    model.fit(train_inputs, y_train, epochs=25, batch_size=32, verbose=0)

    test_predict = model.predict(test_inputs, verbose=0)

    rmse = np.sqrt(mean_squared_error(y_test, test_predict[:, 0]))
    da = calculate_directional_accuracy(y_test, test_predict[:, 0], is_stationary=True)

    print(f"      RMSE: {rmse:.4f} | Directional Accuracy: {da:.2f}%")

    test_dates = df['date'].iloc[-len(y_test):]

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual Returns', color='green', alpha=0.5)
    plt.plot(test_dates, test_predict[:, 0], label='GRU Prediction', color='red', alpha=0.9, linewidth=1.5)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    plt.title(f'{ticker}: Multi-Input GRU (Whale/Retail) [{experiment_name}]\nRMSE={rmse:.4f} | Directional Accuracy={da:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(results_dir, f"GRU_{experiment_name}_{ticker}.png")
    plt.savefig(save_path)
    plt.close()
