# ABOUT
---

> some note for working with AI.

- [Libraries](#libraries)
<br>

# LIBRARIES
---
- [Numpy](#numpy)
- [Matplotlib](#matplotlib)
- [Scikit Learn](#scikit-learn)
- [Pandas](#pandas)

## Numpy
for numeral operation in *Python*<br>
***how to use:***<br>
```Py
import numpy as np
```  
### Methods

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

## Matplotlib
for making an chart or plot with *Python*  
***how to use:***  
```Py 
import matplotlib.pyplot as plt
```  
### Methods

1. for making chart  
	```Py
	plt.plot(x,y)
	```  
2. for show 
	```Py 
	plt.show()
	```  

## Scikit Learn
for machine learning in *Python*  
### Classes

- **linear predict**
    ```Python
    from sklearn.linear_model import LinearRegression
    model_linear = LinearRegression()
    ```  
#### Methods
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
#### Methods:
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
 #### Methods:
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
#### Methods:
1. for fiting  
```Py
label_encoder.fit(x_array)
```  
2. for normalizing  
```Py
x_array_encoded = label_encoder.transform(x_array)
```  

### Methods

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
## Pandas
for reading csv in python
***how to use:***  
```Python 
import pandas as pd
```
### Methods
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
