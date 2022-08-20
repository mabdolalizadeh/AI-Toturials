import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datas = pd.read_csv("CarPrice_Assignment.csv")
inputs = datas[["fueltype", "aspiration", "doornumber", "carbody"]].values
outputs = datas[["price"]].values

i = 0
for fuelType_i in inputs[:, 0]:
    if fuelType_i == "gas":
        inputs[i, 0] = 1
    else:
        inputs[i, 0] = 0
    i += 1

i = 0
for aspiration_i in inputs[:, 1]:
    if aspiration_i == "std":
        inputs[i, 1] = 1
    else:
        inputs[i, 1] = 0
    i += 1

i = 0
for doorNumber_i in inputs[:, 2]:
    if doorNumber_i == "two":
        inputs[i, 2] = 2
    elif doorNumber_i == "four":
        inputs[i, 2] = 4
    i += 1

i = 0
for carBody_i in inputs[:, 3]:
    if carBody_i == "convertible":
        inputs[i, 3] = 0
    elif carBody_i == "hatchback":
        inputs[i, 3] = 1
    elif carBody_i == "sedan":
        inputs[i, 3] = 2
    elif carBody_i == "wagon":
        inputs[i, 3] = 3
    elif carBody_i == "hardtop":
        inputs[i, 3] = 4
    i += 1

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train_norm, y_train)
y_prd = model.predict(x_train_norm)
print(model.score(x_test_norm, y_test))
plt.plot(x_test_norm[:, 0], y_test, 'o')
plt.plot(x_train_norm[:, 0], y_prd, 'o')
plt.show()