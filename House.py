import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ✅ 1. Load the Boston Housing dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.boston_housing.load_data()

print("Training samples:", train_x.shape)
print("Test samples:", test_x.shape)

# ✅ 2. Scale (normalize) the input features for better convergence
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

train=pd.DataFrame(train_x_scaled)
print(train)
print(train.info())
print(train.head())
print(train.columns)
#test_x_scaled


# ✅ 3. Build the Neural Network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[13]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Regression output (predicted price)
])

# ✅ 4. Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ 5. Train the model
history = model.fit(
    train_x_scaled, train_y,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

# ✅ 6. Evaluate on test data
test_loss, test_mae = model.evaluate(test_x_scaled, test_y, verbose=1)
print(f"\nTest MAE (Mean Absolute Error): {test_mae:.2f}")

# ✅ 7. Predict house prices
predictions = model.predict(test_x_scaled[:5])
print("\nPredicted prices:", predictions.flatten())
print("Actual prices:   ", test_y[:5])

# ✅ 8. Optional: Visualize training performance
import matplotlib.pyplot as plt

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Model Training Performance')
plt.show()