from string import digits
from string import ascii_letters
from string import  punctuation
#import keras as ks
#from tensorflow import keras
import tensorflow as ts

help(ts.keras.layers)
ts.keras.models.Sequential([ts.keras.layers.Dense])
def crack4digit():

    for i in digits:
        for j in digits:
            for k in digits:
                for l in digits:
                    print(i,j,k,l)

def crackletter():
    for i in ascii_letters:
        for j in ascii_letters:
            for k in ascii_letters:
                for l in ascii_letters:
                    print(i,j,k,l)

#crackletter()
print(punctuation)
print(len(punctuation))