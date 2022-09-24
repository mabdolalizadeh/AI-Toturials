import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

datas = pd.read_csv("CarPrice_Assignment.csv").values
inputs = datas[:, 1:-1]
output = datas[:, -1].reshape(-1, 1)
np.delete(inputs, 1, 1)

earlyStopper = EarlyStopping(patience=7, verbose=1, restore_best_weights=True)
model = Sequential()
model.add(Dense(units=16, activation='relu'))
model.add(Dense(4, 'relu'))
model.add(Dense(1, 'linear'))
model.compile(optimizer='adam', loss='mse')

encoder = LabelEncoder()
normalize = StandardScaler()

for i in range(1, 8):
    inputs[:, i] = encoder.fit_transform(inputs[:, i])
for j in range(13, 17):
    inputs[:, j] = encoder.fit_transform(inputs[:, j])

x_norm = normalize.fit_transform(inputs)

x_train, x_val_test, y_train, y_val_test = tts(x_norm, output, test_size=0.2)
x_val, x_test, y_val, y_test = tts(x_val_test, y_val_test, test_size=0.33)

result = model.fit(x_train, y_train, epochs=128, validation_data=[x_val, y_val])
y_predicted = model.predict(x_test)

plt.plot(result.history['loss'], label='train')
plt.plot(result.history['val_loss'], label='val')
plt.legend()
plt.show()
