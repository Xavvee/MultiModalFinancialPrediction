import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import get_data, plot_prediction

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run(ticker, results_dir):
    print(f"LSTM (Standard) for {ticker}...")
    prices_series = get_data(ticker)
    dataset = prices_series.values.reshape(-1, 1)

    # Split BEFORE fitting the scaler, then fit on the training split only.
    # fit_transform() on the whole series lets the scaler see the test period's
    # min and max, which is test-set information leaking into training. Test
    # prices outside the training range will scale outside [0, 1] - that is the
    # correct, honest behaviour, not a bug.
    train_size = int(len(dataset) * 0.8)
    train_raw, test_raw = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_raw)
    test_data = scaler.transform(test_raw)

    LOOK_BACK = 15
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])

    test_dates = prices_series.index[-len(y_test_inv[0]):]
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))

    prediction_series = pd.Series(test_predict[:, 0], index=test_dates)
    original_test_series = prices_series.loc[test_dates]
    original_train_series = prices_series.iloc[:train_size]

    save_path = os.path.join(results_dir, f"LSTM_{ticker}.png")
    plot_prediction(original_train_series, original_test_series, prediction_series, f"{ticker}: LSTM Price", save_path, "RMSE", rmse)

    return {'name': 'LSTM (prices)', 'dates': test_dates,
            'y_true': y_test_inv[0], 'y_pred': test_predict[:, 0],
            'is_stationary': False, 'rmse': rmse}