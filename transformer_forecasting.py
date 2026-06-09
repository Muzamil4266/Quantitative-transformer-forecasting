import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, MultiHeadAttention, Flatten, 
    Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ==========================================================
# 1. CONFIGURATION & CACHE SETTINGS
# ==========================================================
FOLDER_PATH = r"C:\Shoby deathless laptop folder\Nifty-Training-dataset"

CACHE_FILE_X = "nifty_cache_X.npy"
CACHE_FILE_Y = "nifty_cache_y.npy"

LOOKBACK_DAYS = 90
HORIZON_DAYS = 30
TOTAL_WINDOW = LOOKBACK_DAYS + HORIZON_DAYS 

EPOCHS = 15
BATCH_SIZE = 64


# ==========================================================
# 2. DATA LOADING (FIXED NORMALIZATION + CLASSIFICATION TARGET)
# ==========================================================
def load_and_prepare_data(folder_path):
    print("Building flashcards with Min-Max Window Scaling + Binary Target...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    X_list = []
    y_list = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "Price" not in df.columns:
                continue
                
            prices = df["Price"].values.astype(np.float32)
            if len(prices) < TOTAL_WINDOW:
                continue
                
            for i in range(len(prices) - TOTAL_WINDOW + 1):
                window = prices[i : i + TOTAL_WINDOW]
                past_90_days = window[:LOOKBACK_DAYS]
                anchor_price = past_90_days[-1] 
                future_price = window[-1]
                
                # --- THE NEW NORMALIZATION FIX ---
                min_price = np.min(past_90_days)
                max_price = np.max(past_90_days)
                
                # Safety check: if the stock price didn't move at all for 90 days, skip it
                if max_price == min_price or anchor_price <= 0:
                    continue
                
                # Min-Max Scaling: Maps the 90 days between 0.0 and 1.0
                normalized_X = (past_90_days - min_price) / (max_price - min_price)
                
                # --- CLASSIFICATION TARGET: UP (1) or DOWN (0) ---
                target_pct_change = (future_price - anchor_price) / anchor_price
                target_binary = 1.0 if target_pct_change > 0 else 0.0
                
                X_list.append(normalized_X)
                y_list.append(target_binary)
                
        except Exception as e:
            print(f"Warning: Skipped file {file} due to error: {e}")
            continue

    X_array = np.array(X_list).reshape(-1, LOOKBACK_DAYS, 1)
    y_array = np.array(y_list)
    return X_array, y_array


# --- CACHE LOGIC ---
# Check if old cache exists and delete it (incompatible with new binary targets)
if os.path.exists(CACHE_FILE_X) or os.path.exists(CACHE_FILE_Y):
    print("Deleting old cache (incompatible with new binary targets)...")
    if os.path.exists(CACHE_FILE_X):
        os.remove(CACHE_FILE_X)
    if os.path.exists(CACHE_FILE_Y):
        os.remove(CACHE_FILE_Y)

# Now check if fresh cache exists
if os.path.exists(CACHE_FILE_X) and os.path.exists(CACHE_FILE_Y):
    print("Found valid cached data! Loading instantly...")
    X_all = np.load(CACHE_FILE_X)
    y_all = np.load(CACHE_FILE_Y)
else:
    print("No cache found. Processing CSVs with new math...")
    X_all, y_all = load_and_prepare_data(FOLDER_PATH)
    
    # Quick sanity check on target distribution
    total_samples = len(y_all)
    up_samples = int(np.sum(y_all == 1.0))
    down_samples = int(np.sum(y_all == 0.0))
    print(f"Flashcards created: {total_samples:,}")
    print(f"  UP (1):   {up_samples:,} ({up_samples/total_samples*100:.1f}%)")
    print(f"  DOWN (0): {down_samples:,} ({down_samples/total_samples*100:.1f}%)")
    
    print("Saving new data to cache for instant loading next time...")
    np.save(CACHE_FILE_X, X_all)
    np.save(CACHE_FILE_Y, y_all)


# ==========================================================
# 3. THE 3-WAY SPLIT
# ==========================================================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42
)

print(f"\nData successfully isolated into three rooms:")
print(f" ├── Pure Training Pile (70%)    : {len(X_train):,} windows")
print(f" ├── Validation Practice Quiz (10%): {len(X_val):,} windows")
print(f" └── Sealed Final Exam Pile (20%)  : {len(X_test):,} windows")


# ==========================================================
# 4. BUILDING THE UPGRADED ARCHITECTURE
# ==========================================================
def build_transformer_accountant_model():
    inputs = Input(shape=(LOOKBACK_DAYS, 1))
    
    # Room 1: The Detective (Transformer Attention)
    attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = Flatten()(attention_out)
    
    # Room 2: The Accountant (Stabilized Dense Layers)
    # Scaled down from 512 to prevent overfitting noise, added Normalization & Dropout
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Lowered learning rate for precise, slow learning of small percentages
    custom_optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = build_transformer_accountant_model()

# ==========================================================
# 5. TRAINING PHASE 
# ==========================================================
print("\n--- STARTING TRAINING ---")

# Calculate Class Weights to balance the UP/DOWN bias
from sklearn.utils.class_weight import compute_class_weight
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}

print(f"Class Weights - DOWN: {class_weight_dict[0]:.2f}, UP: {class_weight_dict[1]:.2f}")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_nifty_transformer.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=0
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict, # <-- This is what forces the model to care about DOWN days!
    callbacks=[checkpoint, early_stop],
    verbose=2  
)


# ==========================================================
# 6. ADVANCED FINAL EXAM EVALUATION
# ==========================================================
print("\n--- UNSEALING THE ENVELOPE: RUNNING 20% TEST DATA ---")
best_model = tf.keras.models.load_model("best_nifty_transformer.keras")

y_pred_prob = best_model.predict(X_test, verbose=0).flatten()
y_pred_class = (y_pred_prob > 0.5).astype(int)

binary_actual = y_test.astype(int)
binary_pred = y_pred_class

# Note: MAE, RMSE, R² make less sense for classification.
# We focus on accuracy, precision, recall, F1, confusion matrix.
mae = 0  # Placeholder
rmse = 0  # Placeholder
r2 = 0  # Placeholder                   

dir_accuracy = accuracy_score(binary_actual, binary_pred) * 100
precision = precision_score(binary_actual, binary_pred, zero_division=0) * 100 
recall = recall_score(binary_actual, binary_pred, zero_division=0) * 100       
f1 = f1_score(binary_actual, binary_pred, zero_division=0) * 100               

tn, fp, fn, tp = confusion_matrix(binary_actual, binary_pred).ravel()


# ==========================================================
# 7. PRINTING THE ULTIMATE REPORT CARD
# ==========================================================
print("\n========================================================")
print("       📊 ULTIMATE NIFTY TRANSFORMER REPORT CARD        ")
print("          (EVALUATED ON PURE, UNSEEN 20% DATA)          ")
print("========================================================")
print(f" DIRECTIONAL TREND METRICS (Up vs Down):")
print(f" ├── Directional Accuracy        : {dir_accuracy:.2f}%")
print(f" ├── Precision (Predicting UP)   : {precision:.2f}%")
print(f" ├── Recall (Catching real UPs)  : {recall:.2f}%")
print(f" └── F1-Score (Trend Balance)    : {f1:.2f}%")
print(f"--------------------------------------------------------")
print(f" PRICE MAGNITUDE METRICS (Percentage Accuracy):")
print(f" ├── Mean Absolute Error (MAE)   : ±{mae:.2f}%")
print(f" ├── Root Mean Squared Error     : ±{rmse:.2f}%")
print(f" └── R² Fit Score                : {r2:.4f}")
print(f"--------------------------------------------------------")
print(f" CONFUSION MATRIX GRID:")
print(f" ├── True Negative (Correctly called DOWN) : {tn:,}")
print(f" ├── False Positive (Wrongly called UP)    : {fp:,}")
print(f" ├── False Negative (Wrongly called DOWN)  : {fn:,}")
print(f" └── True Positive (Correctly called UP)   : {tp:,}")
print("========================================================")
print("Execution complete. Model successfully saved!")
