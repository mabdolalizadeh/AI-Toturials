# ABOUT
---

> some note for working with AI.

# Parts
- [Linears](#linears)
- [Artificial-Neural-Networks](#artificial-neural-networks)
# LINEARs
- [Libraries](#libraries)
- [Notes](#notes)
<br>

## LIBRARIEs
---
- [Numpy](#numpy)
- [Matplotlib](#matplotlib)
- [Scikit Learn](#scikit-learn)
- [Pandas](#pandas)

### Numpy
for numeral operation in *Python*<br>
***how to use:***<br>
```Py
import numpy as np
```  
#### Methods

1. for making array  
	```Py
	np.array(List)
	```   
2. for mean of array  
	```Py
	np.mean(array)
	```  
3. for sum all indexes    
	```Py
	np.sum(array)
	```  

### Matplotlib
for making an chart or plot with *Python*  
***how to use:***  
```Py 
import matplotlib.pyplot as plt
```  
#### Methods

1. for making chart  
	```Py
	plt.plot(x,y)
	```  
2. for show 
	```Py 
	plt.show()
	```  

### Scikit Learn
for machine learning in *Python*  
#### Classes

- **linear predict**
    ```Python
    from sklearn.linear_model import LinearRegression
    model_linear = LinearRegression()
    ```  
##### Methods
1. for fiting your model  
```Py
model_linear.fit(x_array,y_array)
```  
2. for predict  
```Py
y_prd = model_linear.predict(x_array)
```  

- **logestic predict**
    ```Python
    from sklearn.linear_model import LogisticRegression
    model_logestic = LogisticRegression()
    ```  
##### Methods:
1. for fiting your model  
```Py
model_logestic.fit(x_array,y_array)
```  
2. for predict  
```Py
y_prd = model_logestic.predict(x_array)
```  

- **normalize**  
 ```Python
 from sklearn.preprocessing import StandardScaler
 norm = StandardScaler()
 ```  
 ##### Methods:
1. for fiting  
```Py
norm.fit(x_array)
```  
2. for normalizing  
```Py
x_norm = norm.transform(x_array)
```  

- **encoding**
```Python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
```
##### Methods:
1. for fiting  
```Py
label_encoder.fit(x_array)
```  
2. for normalizing  
```Py
x_array_encoded = label_encoder.transform(x_array)
```  
3. for scoring
```Py
from sklearn.metrics import confusion_matrix as cm
cm(y_test, y_test_prd)
```
4. for information of scoring
```Py
from sklearn.metrics import classification_report as cr
cr(y_test, y_test_prd)
```
#### Methods

1. for split datas  
```Python
from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x_array,y_array,test_size=0.2)
```  
2. for mean squarded error(mse)
```Py
from sklearn.metrics import mean_squared_error as mse
mse(x_array)
```
3. for mean absolute error(mae)
```Py
from sklearn.metrics import mean_absolute_error as mae
mae(x_array)
```
### Pandas
for reading csv in python
***how to use:***  
```Python 
import pandas as pd
```
#### Methods
1. reading  
```Py
datas = pd.read_csv("File Adress")
```  
2. show
```Py
datas.head()
```  
3. making array  
```Python
datas_array = datas.values
```  
4. info tell about features
```Py
datas.info()
```  
5. describe tell about mean, std, q1, q2, q3 and etc
```Py
datas.describe()
```
6. count of value
```Py
datas["Name of cloumn"].values_counts()
```

## NOTEs
- **your model must be normal:** for example if you have 2 kind of datas. it must be 50% kind 1 an 50% kind 2.  

# Artificial NEURAL NETWORKS
- [Libraries](#libraries-1)
- [Joblib](#joblib)
- [Open-CV](#open-cv)
- [Notes](#notes-1)

## LIBRARIEs
- [Tensorflow](#transflow)
### Transflow
1. for a model of ANN
```Py
from tensorflow.keras.model import Sequntial
from tensorflow.keras.layers import Dense
model = Sequntial()
#1st layer
model.add(Dense(units = 16, activation = 'relu'))
# units means counts of layers and activation means the method
#2nd layer
model.add(Dense(4, 'relu'))
#final layer
model.add(Dense(1, 'linear'))
#for optimizing
model.compile(optimizer = "adam", loss = "mse")
```
2. for fitting
```Py
result = model.fit(x_train, y_train, epochs = 128, validation_data = [x_val, y_val])
```
3. for getting summary of model
```Py
model.summary()
```
4. for predict
```Py
model.predict()
```
5. for ploting losts
```Py
plt.plot(result.history['loss'], label = 'train')
plt.plot(result.history['val_loss'], label = 'val')
plt.legend()
plt.show()
```
6. for callback or stopping training
```Py
from tensorflow.keras.callbacks import EarlyStopping
early _stopping = EarlyStopping(patience = 5, verbose = 1, restore_best_weights = True)
# patience means after how many no loss stop.
result = model.fit(x_train, y_train, epochs = 128, validation_data = [x_val, y_val], callbacks = early_stopping)
```
7. for saving model
```Py
model.save("Adress/File.h5")
# format of these models are .h5
```
8. for loading another model
```Py
from tensorflow.keras.models import load_model
new_model = load_model("Adress/File.h5")
```  
### Joblib
1. saving normalizer
```Py
from joblib import dump
dump(normalizer, "Adress")
```  
2. loading normalizer
```Py
from joblib import load
normalizer = load("Adress")
```  
### Open CV
> its for image
how to import
```Py
import cv2
```  
how to load  
```Py
img = cv2.imread("Adress")
``` 
to show image  
```Py
cv2.imshow("name", img)
# name is the name of the window of the img
cv2.waitKey(int time)
# int time is number to how many second window wait for waiting at all use 0
``` 
to change color mode
```Py
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
for resizing
```Py
img2 = cv2.resize(img, (Y,X))
```
## NOTEs
### Batch size
How many data goes to layers
### Activation functions
 1. the best for AFs is ***relu function***
 2. ***tanh*** is another one
 3. ***linears*** are another 
 4. you can use ***sigmoid***
 ### Learning rate
 how can you recive the best value(extermum of optimization)
 ### Epoch
 learning rate must be high in first and low in last