
import pandas as pd
import tensorflow as tensor
from tensorflow import keras
import  matplotlib.pyplot as plt
import numpy as np
import math



(train_x,train_y),(test_x,test_x)=keras.datasets.fashion_mnist.load_data()
print(train_x.shape)
print(train_x[0])

dt1= pd.DataFrame(train_x)
dt2= pd.DataFrame(train_y)
print(dt1.shape)  # Should be (60000, 784)
print(dt2.shape)  # Should be (60000, 1) or (60000,)


