import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import get_data, calculate_directional_accuracy

LOOK_BACK = 60
EPOCHS = 15 

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_lstm(data_series):
    dataset = data_series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    train_size = int(len(dataset_scaled) * 0.8)
    train_data, test_data = dataset_scaled[0:train_size, :], dataset_scaled[train_size:len(dataset), :]

    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)

    test_predict = model.predict(X_test)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    
    return y_test_inv[0], test_predict_inv[:,0]

def run(ticker, results_dir):
    print(f"Generowanie Dashboardu z DA...")
    prices = get_data(ticker)
    
    y_true_price, y_pred_price = train_lstm(prices)
    da_price = 0
    if y_true_price is not None:
        da_price = calculate_directional_accuracy(y_true_price, y_pred_price, is_stationary=False)
    
    returns = prices.pct_change().dropna()
    y_true_ret, y_pred_ret = train_lstm(returns)
    da_ret = 0
    if y_true_ret is not None:
        da_ret = calculate_directional_accuracy(y_true_ret, y_pred_ret, is_stationary=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    if y_true_price is not None:
        dates_price = prices.index[-len(y_true_price):]
        ax1.plot(dates_price, y_true_price, label='Rzeczywista Cena', color='green')
        ax1.plot(dates_price, y_pred_price, label='Predykcja', color='red', linestyle='--')
        ax1.set_title(f'{ticker} PRICE Model\n(Lag Effect widoczny, DA={da_price:.2f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if y_true_ret is not None:
        dates_ret = returns.index[-len(y_true_ret):]
        ax2.plot(dates_ret, y_true_ret, label='Rzeczywiste Zmiany', color='green', alpha=0.5)
        ax2.plot(dates_ret, y_pred_ret, label='Predykcja Zmian', color='blue')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_title(f'{ticker} RETURNS Model (Stationary)\n(Prawdziwa zdolność predykcyjna, DA={da_ret:.2f}%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    save_path = os.path.join(results_dir, f"DASHBOARD_COMPARE.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"   ✅ Zapisano dashboard: {save_path}")
    plt.close()