# Cell 1: Debug the data preprocessing and scaling issues
print("=== DEBUGGING DATA PREPROCESSING ===")

# Check original data statistics
print("Original data statistics:")
print("Close price range:", df['Close'].min(), "to", df['Close'].max())
print("Close price mean:", df['Close'].mean())
print("Close price std:", df['Close'].std())

# Check scaled data statistics
print("\nScaled training data statistics:")
print("Scaled close (col 3) range:", scaled_train[:, 3].min(), "to", scaled_train[:, 3].max())
print("Scaled close mean:", scaled_train[:, 3].mean())
print("Scaled close std:", scaled_train[:, 3].std())

# Check if scaling is working correctly
print("\nScaling verification:")
sample_original = train_data['Close'].iloc[:5].values
sample_scaled = scaled_train[:5, 3]
sample_inverse = scaler.inverse_transform(
    np.column_stack([scaled_train[:5, :4], scaled_train[:5, 4:]])
)[:, 3]

print("Original sample:", sample_original)
print("Scaled sample:", sample_scaled)
print("Inverse sample:", sample_inverse)
print("Scaling round-trip error:", np.mean(np.abs(sample_original - sample_inverse)))

# Check target variable distribution
print(f"\nTarget variable (y_train) statistics:")
print(f"y_train range: {y_train.min():.6f} to {y_train.max():.6f}")
print(f"y_train mean: {y_train.mean():.6f}")
print(f"y_train std: {y_train.std():.6f}")

# Check if target has enough variance
if y_train.std() < 0.01:
    print("⚠️  WARNING: Target variable has very low variance!")
    print("This could cause the model to predict constant values.")

# Cell 2: Fix the data preprocessing with better approach
print("\n=== FIXING DATA PREPROCESSING ===")

# Use only Close price for simpler, more reliable scaling
print("Switching to Close-price-only scaling...")

# Separate scaling for features and target
from sklearn.preprocessing import MinMaxScaler

# Scale features (all 5 columns)
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features_train = feature_scaler.fit_transform(train_data)
scaled_features_test = feature_scaler.transform(test_data)

# Scale target (Close price only) separately
target_scaler = MinMaxScaler(feature_range=(0, 1))
close_train = train_data['Close'].values.reshape(-1, 1)
close_test = test_data['Close'].values.reshape(-1, 1)
scaled_target_train = target_scaler.fit_transform(close_train)
scaled_target_test = target_scaler.transform(close_test)

print("Feature scaler fitted on shape:", scaled_features_train.shape)
print("Target scaler fitted on shape:", scaled_target_train.shape)
print("Target scaling range:", scaled_target_train.min(), "to", scaled_target_train.max())
print("Target scaling std:", scaled_target_train.std())

# Recreate sequences with separate scaling
def create_sequences_fixed(features, target, window_size=60):
    """
    Create sequences with properly scaled data
    """
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])  # All features for sequence
        y.append(target[i + window_size])      # Target value
    return np.array(X), np.array(y)

# Create new sequences
X_train_fixed, y_train_fixed = create_sequences_fixed(scaled_features_train, scaled_target_train.flatten(), WINDOW_SIZE)
X_test_fixed, y_test_fixed = create_sequences_fixed(scaled_features_test, scaled_target_test.flatten(), WINDOW_SIZE)

print(f"\nFixed sequences:")
print(f"X_train_fixed shape: {X_train_fixed.shape}")
print(f"y_train_fixed shape: {y_train_fixed.shape}")
print(f"y_train_fixed range: {y_train_fixed.min():.6f} to {y_train_fixed.max():.6f}")
print(f"y_train_fixed std: {y_train_fixed.std():.6f}")

# Verify we have proper variance now
if y_train_fixed.std() > 0.1:
    print("✅ Target variable has good variance now!")
else:
    print("⚠️  Still low variance - need to investigate further")

# Cell 3: Create a simpler, more robust model
print("\n=== CREATING SIMPLER, MORE ROBUST MODEL ===")

def create_simple_robust_model(input_shape, learning_rate=0.001):
    """
    Create a simpler but more robust LSTM model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(64, return_sequences=True, input_shape=input_shape,
             dropout=0.1, recurrent_dropout=0.1),
        
        # Second LSTM layer
        LSTM(32, return_sequences=False,
             dropout=0.1, recurrent_dropout=0.1),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for 0-1 scaled output
    ])
    
    # Use simpler optimizer settings
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Back to MSE for scaled data
        metrics=['mae']
    )
    
    return model

# Create simpler model
simple_model = create_simple_robust_model((WINDOW_SIZE, 5))
simple_model.summary()

print(f"Model parameters: {simple_model.count_params():,}")

# Create simpler callbacks
simple_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

print("Simple model and callbacks created!")

# Cell 4: Train the fixed model
print("\n=== TRAINING FIXED MODEL ===")

# Train with fixed data
history_fixed = simple_model.fit(
    X_train_fixed, y_train_fixed,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=simple_callbacks,
    verbose=1,
    shuffle=False
)

print("Fixed model training completed!")

# Cell 5: Evaluate fixed model
print("\n=== EVALUATING FIXED MODEL ===")

# Make predictions
y_pred_fixed_scaled = simple_model.predict(X_test_fixed, verbose=0)

# Properly inverse transform using target scaler
y_pred_fixed = target_scaler.inverse_transform(y_pred_fixed_scaled.reshape(-1, 1)).flatten()
y_test_fixed = target_scaler.inverse_transform(y_test_fixed.reshape(-1, 1)).flatten()

print(f"Fixed model predictions:")
print(f"Predicted range: {y_pred_fixed.min():.2f} - {y_pred_fixed.max():.2f} INR")
print(f"Actual range: {y_test_fixed.min():.2f} - {y_test_fixed.max():.2f} INR")

# Calculate metrics
metrics_fixed = calculate_metrics(y_test_fixed, y_pred_fixed)

print("\n=== FIXED MODEL PERFORMANCE ===")
print(f"RMSE: {metrics_fixed['RMSE']:.2f} INR")
print(f"MAE: {metrics_fixed['MAE']:.2f} INR")
print(f"R² Score: {metrics_fixed['R2']:.4f}")
print(f"MAPE: {metrics_fixed['MAPE']:.2f}%")
print(f"Directional Accuracy: {metrics_fixed['Directional_Accuracy']:.1%}")

# Quick visualization
plt.figure(figsize=(15, 5))

# Training curves
plt.subplot(1, 3, 1)
plt.plot(history_fixed.history['loss'], label='Training Loss')
plt.plot(history_fixed.history['val_loss'], label='Validation Loss')
plt.title('Fixed Model Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Predictions comparison
plt.subplot(1, 3, 2)
plt.plot(y_test_fixed[:100], label='Actual', linewidth=2)
plt.plot(y_pred_fixed[:100], label='Predicted', linewidth=2)
plt.title('Fixed Model Predictions (First 100)')
plt.xlabel('Time Steps')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot
plt.subplot(1, 3, 3)
plt.scatter(y_test_fixed, y_pred_fixed, alpha=0.6, s=20)
min_val, max_val = min(y_test_fixed.min(), y_pred_fixed.min()), max(y_test_fixed.max(), y_pred_fixed.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.title('Fixed Model Scatter Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare with previous model
print(f"\n=== COMPARISON ===")
print(f"Previous model RMSE: 213.75 INR")
print(f"Fixed model RMSE: {metrics_fixed['RMSE']:.2f} INR")
print(f"Improvement: {((213.75 - metrics_fixed['RMSE']) / 213.75 * 100):.1f}%")
