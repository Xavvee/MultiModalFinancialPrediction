import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate

def create_dual_dataset(price_data, nlp_data, close_price_data, look_back=5):
    """
    Creates sequences for two independent input streams.
    Returns: X_price (OHLCV), X_nlp (FinBERT), Y (Target Close Price)
    """
    X_price, X_nlp, Y = [], [], []
    for i in range(len(price_data) - look_back - 1):
        # 1. Financial sequence (Look_back days, 5 features)
        X_price.append(price_data[i:(i + look_back), :])
        
        # 2. NLP sequence (Look_back days, 768 features)
        X_nlp.append(nlp_data[i:(i + look_back), :])
        
        # 3. Target value (Next day's Close price)
        Y.append(close_price_data[i + look_back, 0])
        
    return np.array(X_price), np.array(X_nlp), np.array(Y)

def run_late_fusion_gru(pkl_path, results_dir):
    print("--- INITIALIZING LATE FUSION GRU MODEL ---")
    
    # 1. Load the dataset containing OHLCV and FinBERT vectors
    df = pd.read_pickle(pkl_path)
    dates = df['date'].values
    
    # Extract 5 financial features (OHLCV)
    financial_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Extract target specifically (Close price) for easier inverse scaling later
    close_price = df[['Close']].values
    
    # Stack NLP features into a proper 2D numpy array
    nlp_features = np.stack(df['finbert_vector'].values)
    
    # 2. Scaling
    # We use two separate scalers to avoid inverse_transform shape issues later
    ohlcv_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_financials = ohlcv_scaler.fit_transform(financial_features)
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(close_price)
    
    # Note: NLP features from FinBERT are typically already small values (-1 to 1), 
    # so we don't scale them to avoid distorting the pre-trained embeddings.

    # 3. Train/Test Split (80% / 20%)
    train_size = int(len(df) * 0.8)
    
    train_fin = scaled_financials[0:train_size, :]
    test_fin = scaled_financials[train_size:len(scaled_financials), :]
    
    train_nlp = nlp_features[0:train_size, :]
    test_nlp = nlp_features[train_size:len(nlp_features), :]
    
    train_target = scaled_target[0:train_size, :]
    test_target = scaled_target[train_size:len(scaled_target), :]

    LOOK_BACK = 5
    EPOCHS = 30
    BATCH_SIZE = 8

    # Create datasets for both streams
    X_train_price, X_train_nlp, y_train = create_dual_dataset(train_fin, train_nlp, train_target, LOOK_BACK)
    X_test_price, X_test_nlp, y_test = create_dual_dataset(test_fin, test_nlp, test_target, LOOK_BACK)

    # ---------------------------------------------------------
    # 4. MODEL ARCHITECTURE: LATE FUSION (BI-MODAL)
    # ---------------------------------------------------------
    
    # STREAM A: Financial Analyst (Processes 5 OHLCV features)
    input_price = Input(shape=(LOOK_BACK, 5), name='Input_OHLCV')
    gru_price = GRU(32, return_sequences=False, name='GRU_Financial')(input_price)
    gru_price = Dropout(0.2)(gru_price)
    
    # STREAM B: NLP Interns (Processes 768 Text features and compresses them)
    input_nlp = Input(shape=(LOOK_BACK, 768), name='Input_FinBERT')
    # The GRU here acts as a powerful dimensionality reduction tool. 
    # It reads 768 dimensions over 'look_back' days and condenses it into 16 core sentiment features.
    gru_nlp = GRU(16, return_sequences=False, name='GRU_NLP')(input_nlp)
    gru_nlp = Dropout(0.2)(gru_nlp)
    
    # Late Fusion Layer
    # We concatenate the 32 financial conclusions with the 16 sentiment conclusions
    fusion_layer = Concatenate(name='Fusion_Concat')([gru_price, gru_nlp])
    
    # Final dense layers to make the ultimate decision
    dense_fusion = Dense(16, activation='relu', name='Dense_Decision')(fusion_layer)
    output = Dense(1, name='Final_Prediction')(dense_fusion)
    
    # Compile the dual-input model
    model = Model(inputs=[input_price, input_nlp], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Print architecture summary for the thesis
    model.summary()
    
    print("\nTraining Late Fusion Model...")
    # Notice we pass a list of inputs [X_price, X_nlp]
    model.fit(
        [X_train_price, X_train_nlp], 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=1
    )

    # 5. Prediction & Evaluation
    print("\nEvaluating Model...")
    test_predict = model.predict([X_test_price, X_test_nlp])
    
    # Inverse transform using our dedicated target scaler
    test_predict_inv = target_scaler.inverse_transform(test_predict)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    print(f"Late Fusion GRU RMSE: {rmse:.2f}")

    # 6. Plotting
    plt.figure(figsize=(10,6))
    test_dates = dates[-len(y_test_inv):]
    
    plt.plot(test_dates, y_test_inv, label='Actual Price (Test)', color='red', linewidth=2)
    plt.plot(test_dates, test_predict_inv, label=f'Late Fusion GRU (RMSE: {rmse:.0f})', color='purple', linewidth=2)
    plt.title('Bi-Modal Late Fusion: GRU (OHLCV) + GRU (FinBERT)')
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "Late_Fusion_GRU_PoC.png"))
    plt.show()

if __name__ == "__main__":
    run_late_fusion_gru('dataset_with_vectors.pkl', 'results')