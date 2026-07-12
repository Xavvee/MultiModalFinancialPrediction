import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), :])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_comparison(pkl_path, results_dir):
    print("--- STARTING COMPARISON TESTS ---")
    df = pd.read_pickle(pkl_path)
    
    prices = df['Close'].values.reshape(-1, 1)
    dates = df['date'].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    nlp_features = np.stack(df['finbert_vector'].values)
    
    # Set 1: Only price (Univariate)
    pure_price_data = scaled_prices
    
    # Set 2: Early Fusion (Price + Text)
    fusion_data = np.hstack((scaled_prices, nlp_features))
    
    train_size = int(len(prices) * 0.8)
    LOOK_BACK = 5
    EPOCHS = 30
    BATCH_SIZE = 8

    # --- MODEL 1: Bare LSTM (Only price) ---
    print("\nTraining Model 1: Bare LSTM...")
    train_pure, test_pure = pure_price_data[0:train_size, :], pure_price_data[train_size:len(pure_price_data), :]
    X_train_pure, y_train_pure = create_dataset(train_pure, LOOK_BACK)
    X_test_pure, y_test_pure = create_dataset(test_pure, LOOK_BACK)

    model_pure = Sequential()
    model_pure.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
    model_pure.add(Dense(1))
    model_pure.compile(loss='mean_squared_error', optimizer='adam')
    model_pure.fit(X_train_pure, y_train_pure, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    predict_pure = scaler.inverse_transform(model_pure.predict(X_test_pure))

    # --- MODEL 2: Early Fusion (Price + FinBERT) ---
    print("Training Model 2: Early Fusion...")
    train_fusion, test_fusion = fusion_data[0:train_size, :], fusion_data[train_size:len(fusion_data), :]
    X_train_fus, y_train_fus = create_dataset(train_fusion, LOOK_BACK)
    X_test_fus, y_test_fus = create_dataset(test_fusion, LOOK_BACK)

    model_fus = Sequential()
    model_fus.add(LSTM(50, input_shape=(LOOK_BACK, fusion_data.shape[1])))
    model_fus.add(Dropout(0.2))
    model_fus.add(Dense(1))
    model_fus.compile(loss='mean_squared_error', optimizer='adam')
    model_fus.fit(X_train_fus, y_train_fus, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    predict_fusion = scaler.inverse_transform(model_fus.predict(X_test_fus))

    # --- CALCULATING ERRORS ---
    y_test_inv = scaler.inverse_transform(y_test_pure.reshape(-1, 1))
    rmse_pure = np.sqrt(mean_squared_error(y_test_inv, predict_pure))
    rmse_fusion = np.sqrt(mean_squared_error(y_test_inv, predict_fusion))

    print(f"\nFinal Results:")
    print(f"RMSE Bare LSTM: {rmse_pure:.2f}")
    print(f"RMSE Early Fusion: {rmse_fusion:.2f}")

    # --- PLOTTING THE GRAPH ---
    plt.figure(figsize=(12,7))
    test_dates = dates[-len(y_test_inv):]
    
    plt.plot(test_dates, y_test_inv, label='Actual Price (Test)', color='red', linewidth=2.5)
    plt.plot(test_dates, predict_pure, label=f'Bare LSTM (RMSE: {rmse_pure:.0f})', color='green', linestyle='dashed', linewidth=2)
    plt.plot(test_dates, predict_fusion, label=f'Early Fusion (RMSE: {rmse_fusion:.0f})', color='blue', alpha=0.7, linewidth=2)
    
    plt.title('Model Comparison: Bare LSTM vs Early Fusion (PoC Sample)')
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "Model_Comparison_PoC.png"))
    plt.show()

if __name__ == "__main__":
    run_comparison('dataset_with_vectors.pkl', 'results')