import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from utils import calculate_directional_accuracy

def create_multivariate_dataset(dataset_features, dataset_target, look_back=1):
    """
    Creates a dataset for time series prediction.
    Returns features (X) and continuous target (y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset_features) - look_back - 1):
        dataX.append(dataset_features[i:(i + look_back), :])
        dataY.append(dataset_target[i + look_back])
    return np.array(dataX), np.array(dataY)

def run(ticker, results_dir, dataset_pkl='data/old_dataset/processed/full_dataset_weighted.pkl'):
    # ==========================================
    # EXPERIMENT TRACKING: Change this name before every new idea!
    experiment_name = "V0_Regression_RMSE"
    # ==========================================
    
    print(f"GRU Multi-Modal [{experiment_name}] for {ticker}...")
    
    if not os.path.exists(dataset_pkl):
        print(f"      ERROR: Cannot find {dataset_pkl}. Skipping GRU Multi-modal.")
        return

    df = pd.read_pickle(dataset_pkl)
    
    if ticker != "BTC-USD":
        print(f"      INFO: NLP dataset is specifically for BTC-USD. Skipping for {ticker}.")
        return

    # DATA TRUNCATION: Cut off the stagnant years 2014-2016
    df = df[df['date'] >= '2017-01-01'].copy()

    feature_columns = [
        'daily_return', 
        'momentum_3d', 
        'volatility_7d', 
        'finbert_sentiment', 
        'roberta_sentiment'
    ]
    
    df = df.dropna(subset=feature_columns)
    
    # Input features (X) and target (raw daily return)
    dataset_features = df[feature_columns].values
    target_returns = df['daily_return'].values

    # Split into train and test sets (80/20 ratio) BEFORE scaling
    train_size = int(len(dataset_features) * 0.8)

    train_features_raw = dataset_features[0:train_size, :]
    train_target = target_returns[0:train_size]

    test_features_raw = dataset_features[train_size:len(dataset_features), :]
    test_target = target_returns[train_size:len(target_returns)]

    # Fit the scaler on the training split only, then transform both splits,
    # so test-set statistics never leak into training.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = scaler.fit_transform(train_features_raw)
    test_features = scaler.transform(test_features_raw)

    LOOK_BACK = 14 
    
    X_train, y_train = create_multivariate_dataset(train_features, train_target, LOOK_BACK)
    X_test, y_test = create_multivariate_dataset(test_features, test_target, LOOK_BACK)

    # Build the Neural Network Architecture
    model = Sequential()
    
    # Layer 1: GRU
    model.add(GRU(64, return_sequences=True, input_shape=(LOOK_BACK, len(feature_columns))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Layer 2: GRU
    model.add(GRU(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output Layer - CHANGED TO REGRESSION (Linear activation)
    model.add(Dense(1))
    
    # CHANGE LOSS FUNCTION TO MEAN SQUARED ERROR
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("      Training GRU model on GPU for Regression...")
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    # Predictions are now actual return values
    test_predict = model.predict(X_test, verbose=0)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, test_predict[:, 0]))
    
    # Calculate Directional Accuracy
    da = calculate_directional_accuracy(y_test, test_predict[:, 0], is_stationary=True)
    
    print(f"      RMSE: {rmse:.4f} | Directional Accuracy: {da:.2f}%")

    # Plotting
    test_dates = df['date'].iloc[-len(y_test):]
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual Returns', color='green', alpha=0.5)
    plt.plot(test_dates, test_predict[:, 0], label='GRU Prediction', color='red', alpha=0.9, linewidth=1.5)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    plt.title(f'{ticker}: Multi-Modal GRU [{experiment_name}]\nRMSE={rmse:.4f} | Directional Accuracy={da:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SAVE WITH EXPERIMENT NAME
    save_path = os.path.join(results_dir, f"GRU_{experiment_name}_{ticker}.png")
    plt.savefig(save_path)
    plt.close()

    return {'name': 'GRU Dual-Stream', 'dates': pd.DatetimeIndex(test_dates),
            'y_true': y_test, 'y_pred': test_predict[:, 0],
            'is_stationary': True, 'rmse': rmse}