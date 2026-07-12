import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate

def create_univariate_dataset(dataset, look_back=1):
    """For Pure LSTM and Early Fusion"""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def create_bimodal_dataset(price_data, nlp_data, target_data, look_back=1):
    """For Late Fusion GRU (Multi-Input)"""
    X_price, X_nlp, Y = [], [], []
    for i in range(len(price_data) - look_back - 1):
        X_price.append(price_data[i:(i + look_back), :])
        X_nlp.append(nlp_data[i:(i + look_back), :])
        Y.append(target_data[i + look_back, 0])
    return np.array(X_price), np.array(X_nlp), np.array(Y)

def run_master_comparison(pkl_path, results_dir):
    print("--- STARTING MASTER ABLATION STUDY ---")
    
    # 1. LOAD DATA
    df = pd.read_pickle(pkl_path)
    dates = df['date'].values
    
    close_price = df[['Close']].values
    ohlcv_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    nlp_features = np.stack(df['finbert_vector'].values)
    
    # 2. SCALERS
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = close_scaler.fit_transform(close_price)
    
    ohlcv_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_ohlcv = ohlcv_scaler.fit_transform(ohlcv_features)
    
    # 3. DATA SPLITS (80/20)
    train_size = int(len(df) * 0.8)
    LOOK_BACK = 5
    EPOCHS = 30
    BATCH_SIZE = 8
    
    # --- MODEL 1: PURE LSTM (Baseline) ---
    print("\n[1/3] Training Pure LSTM (Close Price Only)...")
    X_train_m1, y_train_m1 = create_univariate_dataset(scaled_close[:train_size], LOOK_BACK)
    X_test_m1, y_test_m1 = create_univariate_dataset(scaled_close[train_size:], LOOK_BACK)
    
    model_1 = Sequential([
        LSTM(50, input_shape=(LOOK_BACK, 1)),
        Dense(1)
    ])
    model_1.compile(loss='mse', optimizer='adam')
    model_1.fit(X_train_m1, y_train_m1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred_m1 = close_scaler.inverse_transform(model_1.predict(X_test_m1, verbose=0))
    
    # --- MODEL 2: EARLY FUSION LSTM (Close + 768 FinBERT) ---
    print("[2/3] Training Early Fusion LSTM (Modality Dominance Check)...")
    early_fusion_data = np.hstack((scaled_close, nlp_features))
    X_train_m2, y_train_m2 = create_univariate_dataset(early_fusion_data[:train_size], LOOK_BACK)
    X_test_m2, y_test_m2 = create_univariate_dataset(early_fusion_data[train_size:], LOOK_BACK)
    
    model_2 = Sequential([
        LSTM(50, input_shape=(LOOK_BACK, early_fusion_data.shape[1])),
        Dropout(0.2),
        Dense(1)
    ])
    model_2.compile(loss='mse', optimizer='adam')
    model_2.fit(X_train_m2, y_train_m2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred_m2 = close_scaler.inverse_transform(model_2.predict(X_test_m2, verbose=0))
    
    # --- MODEL 3: LATE FUSION GRU (OHLCV + Compressed FinBERT) ---
    print("[3/3] Training Late Fusion GRU (Target Architecture)...")
    X_train_m3_price, X_train_m3_nlp, y_train_m3 = create_bimodal_dataset(
        scaled_ohlcv[:train_size], nlp_features[:train_size], scaled_close[:train_size], LOOK_BACK)
    X_test_m3_price, X_test_m3_nlp, y_test_m3 = create_bimodal_dataset(
        scaled_ohlcv[train_size:], nlp_features[train_size:], scaled_close[train_size:], LOOK_BACK)

    input_price = Input(shape=(LOOK_BACK, 5), name='Input_OHLCV')
    gru_price = GRU(32)(input_price)
    
    input_nlp = Input(shape=(LOOK_BACK, 768), name='Input_FinBERT')
    gru_nlp = GRU(16)(input_nlp)
    
    fusion = Concatenate()([gru_price, gru_nlp])
    dense_out = Dense(16, activation='relu')(fusion)
    output = Dense(1)(dense_out)
    
    model_3 = Model(inputs=[input_price, input_nlp], outputs=output)
    model_3.compile(loss='mse', optimizer='adam')
    model_3.fit([X_train_m3_price, X_train_m3_nlp], y_train_m3, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred_m3 = close_scaler.inverse_transform(model_3.predict([X_test_m3_price, X_test_m3_nlp], verbose=0))
    
    # --- EVALUATION & PLOTTING ---
    y_true = close_scaler.inverse_transform(y_test_m1.reshape(-1, 1))
    
    rmse_m1 = np.sqrt(mean_squared_error(y_true, pred_m1))
    rmse_m2 = np.sqrt(mean_squared_error(y_true, pred_m2))
    rmse_m3 = np.sqrt(mean_squared_error(y_true, pred_m3))
    
    print("\n--- FINAL RESULTS ---")
    print(f"1. Pure LSTM RMSE:       {rmse_m1:.2f}")
    print(f"2. Early Fusion RMSE:    {rmse_m2:.2f}")
    print(f"3. Late Fusion GRU RMSE: {rmse_m3:.2f}")

    plt.figure(figsize=(14, 8))
    test_dates = dates[-len(y_true):]
    
    plt.plot(test_dates, y_true, label='Actual Price', color='black', linewidth=3)
    plt.plot(test_dates, pred_m1, label=f'Baseline: Pure LSTM (RMSE: {rmse_m1:.0f})', color='green', linestyle='dashed', linewidth=2)
    plt.plot(test_dates, pred_m2, label=f'Failed: Early Fusion (RMSE: {rmse_m2:.0f})', color='red', alpha=0.6, linewidth=2)
    plt.plot(test_dates, pred_m3, label=f'Target: Late Fusion GRU (RMSE: {rmse_m3:.0f})', color='blue', linewidth=2.5)
    
    plt.title('Ablation Study: Architecture Evolution (Proof of Concept)')
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "Master_Comparison_PoC.png"))
    plt.show()

if __name__ == "__main__":
    run_master_comparison('dataset_with_vectors.pkl', 'results')