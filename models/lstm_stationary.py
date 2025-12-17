import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils import get_data, calculate_directional_accuracy

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run(ticker, results_dir):
    print(f"LSTM (Returns) dla {ticker}...")
    prices = get_data(ticker)
    returns = prices.pct_change().dropna()
    
    dataset = returns.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    train_size = int(len(dataset_scaled) * 0.8)
    train_data, test_data = dataset_scaled[0:train_size, :], dataset_scaled[train_size:len(dataset), :]

    LOOK_BACK = 60
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    test_predict = model.predict(X_test)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])

    test_dates = returns.index[-len(y_test_inv[0]):]
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict_inv[:,0]))
    
    da = calculate_directional_accuracy(y_test_inv[0], test_predict_inv[:,0], is_stationary=True)
    print(f"      Wynik RMSE: {rmse:.6f} | Dir Acc: {da:.2f}%")

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_inv[0], label='Rzeczywiste Zmiany', color='green', alpha=0.6)
    plt.plot(test_dates, test_predict_inv[:,0], label='Predykcja', color='red', alpha=0.8)
    
    plt.title(f'{ticker}: LSTM Stationary Forecast\nRMSE={rmse:.4f} | Directional Accuracy={da:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(results_dir, f"LSTM_Returns_{ticker}.png")
    plt.savefig(save_path)
    plt.close()