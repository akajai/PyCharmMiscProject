import pandas as pd
import tensorflow as tf
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

(train_images, train_labels),(test_images, test_labels),  = tf.keras.datasets.mnist.load_data()

rows=int(math.sqrt(len(test_images)))
cols=int(math.sqrt(len(test_images)))
rows=10
cols=20

plt.figure(figsize=(10,10))

for i in range(rows*cols):
  ax = plt.subplot(rows, cols, i + 1)
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  plt.xlabel(test_labels[i])
  plt.xticks([])
  plt.yticks([])


plt.show()


train_images = train_images / 255.0
test_images = test_images / 255.0
#plt.show(test_images)

modelmnist=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                      tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                       tf.keras.layers.Dense(10,activation=tf.nn.softmax)

                                      ])
modelmnist.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
modelmnist.fit(train_images,train_labels,epochs=30,callbacks=[myCallback()])
evaluate=modelmnist.evaluate(test_images,test_labels)
print(evaluate)

