import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

datas = pd.read_csv("CarPrice_Assignment.csv").values
inputs = datas[:, 1:-1]
output = datas[:, -1].reshape(1, -1)
earlyStopper = EarlyStopping(patience=7, verbose=1, restore_best_weights=True)
model = Sequential()
model.add(Dense(units=16, activation='relu'))
model.add(Dense(4, 'relu'))
model.add(Dense(1, 'linear'))
model.compile(optimizer='adam', loss='mse')

encoder = LabelEncoder()
normalize = StandardScaler()

x_encoded = encoder.fit_transform(inputs)
x_norm = normalize.fit_transform(x_encoded)

x_train, y_train, x_test, y_test = tts(x_norm, output, test_size=0.2)

