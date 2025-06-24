# Cell 1: Import additional libraries for improved model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Cell 2: Create improved LSTM architecture
def create_improved_lstm_model(input_shape, learning_rate=0.001):
    """
    Create an improved LSTM model with better architecture
    """
    model = Sequential([
        # First LSTM layer - return sequences for stacking
        LSTM(100, return_sequences=True, input_shape=input_shape,
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer - return sequences for stacking
        LSTM(80, return_sequences=True,
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Third LSTM layer - no return sequences (final layer)
        LSTM(60, return_sequences=False,
             dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers for final prediction
        Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(25, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='linear')  # Linear for regression
    ])
    
    # Compile with custom optimizer
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Gradient clipping
    model.compile(
        optimizer=optimizer,
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae', 'mse']
    )
    
    return model

# Create the improved model
improved_model = create_improved_lstm_model((WINDOW_SIZE, 5))
improved_model.summary()

# Cell 3: Setup advanced callbacks for better training
def create_callbacks(model_name='improved_lstm'):
    """
    Create callbacks for better training control
    """
    callbacks = [
        # Early stopping with more patience
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            cooldown=5
        ),
        
        # Save best model
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

# Create callbacks
callbacks = create_callbacks('improved_lstm_reliance')
print("Callbacks created successfully!")

# Cell 4: Train the improved model
print("Starting training...")
print(f"Training data shape: {X_train.shape}")
print(f"Validation split will create ~{int(len(X_train) * 0.2)} validation samples")

# Train the improved model
history_improved = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,  # More epochs with better early stopping
    batch_size=16,  # Smaller batch size for more stable training
    callbacks=callbacks,
    verbose=1,
    shuffle=False  # Important for time series!
)

print("Training completed!")

# Cell 5: Visualize improved training history
plt.figure(figsize=(15, 5))

# Plot 1: Loss curves
plt.subplot(1, 3, 1)
plt.plot(history_improved.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history_improved.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss (Improved)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: MAE curves
plt.subplot(1, 3, 2)
plt.plot(history_improved.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history_improved.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('Model MAE (Improved)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Learning rate (if available)
plt.subplot(1, 3, 3)
if 'lr' in history_improved.history:
    plt.plot(history_improved.history['lr'], linewidth=2, color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
else:
    plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print training summary
print(f"\nTraining Summary:")
print(f"Total epochs: {len(history_improved.history['loss'])}")
print(f"Final training loss: {history_improved.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history_improved.history['val_loss'][-1]:.6f}")
print(f"Best validation loss: {min(history_improved.history['val_loss']):.6f}")
print(f"Final training MAE: {history_improved.history['mae'][-1]:.6f}")
print(f"Final validation MAE: {history_improved.history['val_mae'][-1]:.6f}")

# Cell 6: Load best model and make predictions
from tensorflow.keras.models import load_model

# Load the best saved model
best_model = load_model('improved_lstm_reliance_best.h5')
print("Best model loaded successfully!")

# Make predictions on test set
y_pred_improved = best_model.predict(X_test, verbose=0)
print(f"Predictions shape: {y_pred_improved.shape}")

# Inverse transform predictions and actual values
def inverse_transform_predictions(y_pred, y_true, scaler):
    """
    Helper function to inverse transform predictions
    """
    # Create padded arrays for inverse scaling
    y_pred_padded = np.zeros((y_pred.shape[0], 5))
    y_true_padded = np.zeros((y_true.shape[0], 5))
    
    # Fill the Close price column (index 3)
    y_pred_padded[:, 3] = y_pred.flatten()
    y_true_padded[:, 3] = y_true
    
    # Inverse transform
    y_pred_actual = scaler.inverse_transform(y_pred_padded)[:, 3]
    y_true_actual = scaler.inverse_transform(y_true_padded)[:, 3]
    
    return y_pred_actual, y_true_actual

# Get actual price predictions
y_pred_inr, y_test_inr = inverse_transform_predictions(y_pred_improved, y_test, scaler)

print(f"Prediction range: {y_pred_inr.min():.2f} - {y_pred_inr.max():.2f} INR")
print(f"Actual range: {y_test_inr.min():.2f} - {y_test_inr.max():.2f} INR")

# Cell 7: Comprehensive model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    actual_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

# Calculate metrics for improved model
metrics_improved = calculate_metrics(y_test_inr, y_pred_inr)

print("=== IMPROVED MODEL PERFORMANCE ===")
print(f"RMSE: {metrics_improved['RMSE']:.2f} INR")
print(f"MAE: {metrics_improved['MAE']:.2f} INR")
print(f"R² Score: {metrics_improved['R2']:.4f}")
print(f"MAPE: {metrics_improved['MAPE']:.2f}%")
print(f"Directional Accuracy: {metrics_improved['Directional_Accuracy']:.1%}")

# Calculate relative improvement if you have previous model results
print(f"\nModel Complexity:")
print(f"Total Parameters: {best_model.count_params():,}")
print(f"Trainable Parameters: {sum([np.prod(w.shape) for w in best_model.trainable_weights]):,}")

# Cell 8: Advanced visualization of results
plt.figure(figsize=(20, 12))

# Plot 1: Full prediction comparison
plt.subplot(2, 3, 1)
plt.plot(y_test_inr, label='Actual', linewidth=2, alpha=0.8)
plt.plot(y_pred_inr, label='Predicted', linewidth=2, alpha=0.8)
plt.title('Actual vs Predicted Prices (Full Test Set)')
plt.xlabel('Time Steps')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Zoomed view (first 100 points)
plt.subplot(2, 3, 2)
zoom_points = min(100, len(y_test_inr))
plt.plot(y_test_inr[:zoom_points], label='Actual', linewidth=2, marker='o', markersize=3)
plt.plot(y_pred_inr[:zoom_points], label='Predicted', linewidth=2, marker='s', markersize=3)
plt.title(f'Zoomed View (First {zoom_points} Points)')
plt.xlabel('Time Steps')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Prediction errors
plt.subplot(2, 3, 3)
errors = y_pred_inr - y_test_inr
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (INR)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Plot 4: Scatter plot
plt.subplot(2, 3, 4)
plt.scatter(y_test_inr, y_pred_inr, alpha=0.6, s=20)
# Perfect prediction line
min_val, max_val = min(y_test_inr.min(), y_pred_inr.min()), max(y_test_inr.max(), y_pred_inr.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.title('Predicted vs Actual Scatter Plot')
plt.grid(True, alpha=0.3)

# Plot 5: Residuals over time
plt.subplot(2, 3, 5)
plt.plot(errors, linewidth=1)
plt.title('Residuals Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Residual (INR)')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Plot 6: Percentage errors
plt.subplot(2, 3, 6)
pct_errors = ((y_pred_inr - y_test_inr) / y_test_inr) * 100
plt.hist(pct_errors, bins=50, alpha=0.7, edgecolor='black')
plt.title('Percentage Error Distribution')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print additional statistics
print(f"\nError Statistics:")
print(f"Mean Error: {np.mean(errors):.2f} INR")
print(f"Std Error: {np.std(errors):.2f} INR")
print(f"Max Error: {np.max(np.abs(errors)):.2f} INR")
print(f"95th Percentile Error: {np.percentile(np.abs(errors), 95):.2f} INR")
