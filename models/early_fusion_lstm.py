import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def create_multivariate_dataset(dataset, look_back=1):
    """Creates time windows for multiple features at once.
    dataset shape: (number_of_days, 769) -> 1 price + 768 NLP features.
    Returns X with 769 features and y with only the price."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # Bierzemy całe okno (look_back) i wszystkie kolumny (cechy)
        dataX.append(dataset[i:(i + look_back), :])
        # Jako target (Y) bierzemy tylko kolumnę indeks 0, czyli znormalizowaną cenę z następnego dnia
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run_early_fusion(pkl_path, results_dir):
    print("Initializing Early Fusion Model (Price + FinBERT)...")
    
    # 1. Loading data with vectors
    df = pd.read_pickle(pkl_path)
    
    # Keeping only what we are interested in: Date, Close Price, and FinBERT vectors
    prices = df['Close'].values.reshape(-1, 1)
    dates = df['date'].values
    
    # 2. Price scaling (MinMaxScaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # 3. Extracting FinBERT vectors (this is a list of arrays, we need to make it one large matrix)
    nlp_features = np.stack(df['finbert_vector'].values) # Shape: (number_of_days, 768)
    
    # 4. DATA FUSION (Early Fusion): Combining the scaled price with the 768 NLP features
    # Shape after concatenation: (number_of_days, 769)
    combined_data = np.hstack((scaled_prices, nlp_features))
    
    print(f"Shape of combined data: {combined_data.shape} (Days, Features)")

    # 5. Splitting into training and testing sets (80% / 20%)
    train_size = int(len(combined_data) * 0.8)
    train_data = combined_data[0:train_size, :]
    test_data = combined_data[train_size:len(combined_data), :]

    LOOK_BACK = 5 # For a 100-day window, it's better to reduce look_back to have enough data for training
    
    X_train, y_train = create_multivariate_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_multivariate_dataset(test_data, LOOK_BACK)

    print(f"Shape of X_train: {X_train.shape} -> (Batch, LookBack, Features)")

    # 6. Building the Neural Network
    model = Sequential()
    model.add(LSTM(50, input_shape=(LOOK_BACK, combined_data.shape[1]), return_sequences=False))
    model.add(Dropout(0.2)) # Small dropout to prevent overfitting given the small dataset
    model.add(Dense(1)) # On the output we want only one value (the price)
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("Training the model (Early Fusion)...")
    model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=1)

    # 7. Prediction and Evaluation
    test_predict = model.predict(X_test)
    
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    print(f"Final RMSE: {rmse}")

    plt.figure(figsize=(10,6))
    test_dates = dates[-len(y_test_inv):]
    
    plt.plot(test_dates, y_test_inv, label='Actual Price (Test)', color='red')
    plt.plot(test_dates, test_predict_inv, label='Prediction (Early Fusion)', color='blue')
    plt.title(f'BTC Prediction - Early Fusion (LSTM + FinBERT) | RMSE: {rmse:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "Early_Fusion_PoC.png"))
    plt.show()

if __name__ == "__main__":
    pkl_file = 'dataset_with_vectors.pkl'
    results_directory = 'results'
    
    run_early_fusion(pkl_file, results_directory)