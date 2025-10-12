import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sqlalchemy.dialects.mssql.information_schema import columns
from tensorflow import keras

mnist=keras.datasets.mnist.load_data()
print(mnist)

(boston_train_data, boston_train_labels), (boston_test_data, boston_test_labels)=keras.datasets.boston_housing.load_data()
(california_train_data, california_train_labels), (california_test_data, california_test_labels)=keras.datasets.california_housing.load_data()

df_boston=pd.DataFrame(boston_test_labels,columns=columns['median_price']
)
print(df_boston.head())



# Load Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize data (important for training stability)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build model
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Train
model1.fit(train_images, train_labels, epochs=5)

# Evaluate
test_loss, test_acc = model1.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# ---- Pick one test image ----
index = 10   # change index to test different images
img = test_images[index]
true_label = test_labels[index]

# Model prediction (need batch dimension)
prediction = model1.predict(np.expand_dims(img, axis=0))
predicted_label = np.argmax(prediction)

# ---- Display ----
plt.figure()
plt.imshow(img, cmap='gray')
plt.title(f"True: {class_names[true_label]} | Predicted: {class_names[predicted_label]}")
plt.axis('off')
plt.show()