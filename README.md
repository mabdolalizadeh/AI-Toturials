# ABOUT
---

> some note for working with AIs.

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
`import numpy as np`<br>
### Methods
-
1. for making array  
	`np.array(List)`   
2. for mean of array  
	`np.mean(array)`
3. for sum all indexes  
	`np.sum(array)`

## Matplotlib
for making an chart or plot with *Python*  
***how to use:***  
`import matplotlib.pyplot as plt`  
### Methods
-
1. for making chart  
	`plt.plot(x,y)`   
2. for show 
	`plt.show()`

## Scikit Learn
for machine learning in *Python*  
### Classes
-
- **linear predict**
    ```Python
    from sklearn.linear_model import LinearRegression
    model_linear = LinearRegression()
    ```  
#### Methods
	1. for fiting your model  
`model_linear.fit(x_array,y_array)`  
	2. for predict  
`y_prd = model_linear.predict(x_array)`  

- **logestic predict**
    ```Python
    from sklearn.linear_model import LogisticRegression
    model_logestic = LogisticRegression()
    ```  
#### Methods:
	1. for fiting your model  
`model_logestic.fit(x_array,y_array)`  
	2. for predict  
`y_prd = model_logestic.predict(x_array)`  

- **normalize**  
 ```Python
 from sklearn.preprocessing import StandardScaler
 norm = StandardScaler()
 ```  
 #### Methods:
    1. for fiting  
   `norm.fit(x_array)`  
    2. for normalizing  
    `x_norm = norm.transform(x_array)`  

- **encoding**
```Python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
```
#### Methods:
    1. for fiting  
   `label_encoder.fit(x_array)`  
    2. for normalizing  
    `x_array_encoded = label_encoder.transform(x_array)`  

### Methods
-
    1. for split datas  
    ```Python
    from sklearn.model_selection import train_test_split
    x_train, y_train, x_test, y_test = train_test_split(x_array,y_array,test_size=0.2)
    ```  

## Pandas
for reading csv in python
***how to use:***  
`import pandas as pd`  
### Methods
    1. reading
    `datas = pd.read_csv("File Adress")`
    2. show
    `datas.head()`
    3. making array
    `datas_array = datas.values`