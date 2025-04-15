# Python 3.5 is required (주석은 PDF 내용대로)
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# read the file containing the pima indians diabetes data set
data = pd.read_csv('./diabetes.csv', sep=',') # diabetes.csv 파일이 같은 폴더에 있어야 함

print("\ndata.head(): \n", data.head())

# describe the columns of the data set
print("\ndata.describe(): \n", data.describe())

# see if the data set has null values
print("\ndata.info():")
data.info()

print("\n\nStep 2 Prepare the data for the model building")
# extract the X and y from the imported data
X = data.values[:, 0:8]
y = data.values[:, 8]

# use MinMaxScaler to fit a scaler object
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the test set into the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 Create and train the model")
# create the model
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs, output)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0) # verbose=0이면 학습 과정 출력 안 함

# summarize history for loss and accuracy as a function of the epochs
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# Evaluate on a few test samples
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

print("\nExporting SavedModels: ")

# Keras 모델 저장
model.save('pima_model.keras') # 모델을 .keras 형식으로 저장

# 모델 로드 (확인용)
model_loaded = keras.models.load_model('pima_model.keras')

# evaluate model (확인용)
print("\nnp.round(model_loaded.predict(X_new), 2): \n",
      np.round(model_loaded.predict(X_new), 2))

print("\nModel saved as pima_model.keras")