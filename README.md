# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.

2. 6Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE,MAE and RMSE.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Preethika N
RegisterNumber:  212223040130

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(2)
df.tail(4)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='violet')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
/*
```

## Output:

Data set

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/81455bb9-b8a6-4774-8b73-3a3336e91d68)

Head values

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/db447040-b591-4951-aad1-f6bcb5b85981)

Tail values

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/56289474-aa6a-4b3c-8276-6fc0ea104317)

X values

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/a97575e8-220b-4b6c-854c-92eec7fda303)

Y values

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/82216034-f46f-4566-8bdb-579fe3acbb7b)

Prediction values

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/08881446-e3f0-4189-bc10-5248678ac6cd)

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/abf710ea-0ff9-4d1c-939d-8d55f5d1d8b1)

MSE,MAE and RMSE

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/283dab67-6e9e-4715-b820-3b4f611c71b1)


Training set

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/ba23a23a-9e64-4c6b-b511-f16f895dfb89)

Testing test

![image](https://github.com/preethi2831/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155142246/7880c5b3-764c-448e-8573-e08ce87a4a04)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
