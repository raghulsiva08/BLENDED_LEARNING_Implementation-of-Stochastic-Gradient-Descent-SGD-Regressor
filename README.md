# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and perform initial inspection; remove irrelevant columns and convert categorical variables into numerical form using one-hot encoding.
2. Separate the dataset into independent features (X) and target variable (y), and apply standard scaling to normalize the data.
3. Split the scaled data into training and testing sets using an 80:20 ratio.
4. Initialize the Stochastic Gradient Descent (SGD) Regressor and train the model using the training data.
5. Predict car prices on the test data, evaluate the model using MSE, MAE, and R² score, and visualize actual versus predicted values.


## Program:
~~~
/*
Program to implement SGD Regressor for linear regression.
Developed by: RAGHUL.S
RegisterNumber:  212225040325
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

#load the data set
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#data preprocessing,dropping the unnecessary coloumn and handling the catergorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data = pd.get_dummies(data, drop_first=True)

#splitting the data 
X=data.drop('price', axis=1)
Y=data['price']

scaler = StandardScaler()
X=scaler.fit_transform(X)
Y=scaler.fit_transform(np.array(Y).reshape(-1, 1))

#splitting the dataset into training and tests
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#create sdg regressor model
sgd_model= SGDRegressor(max_iter=1000, tol=1e-3)

#fiting the model to training data
sgd_model.fit(X_train, Y_train)

#making predictions
y_pred = sgd_model.predict(X_test)

#evaluating model performance
mse = mean_squared_error(Y_test, y_pred)
r2=r2_score(Y_test,y_pred)
mae= mean_absolute_error(Y_test, y_pred)

#print evaluation metrics
print('Name:RAGHUL.S')
print('Reg no: 212225040325')
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-Squared Score:",r2)

#print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#visualising actual vs predicted prices
plt.scatter(Y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(Y_test),max(Y_test)],[min(Y_test),max(Y_test)],color='red')
plt.show()
~~~
## Output:
![WhatsApp Image 2026-02-12 at 12 26 39 AM](https://github.com/user-attachments/assets/12d8ca52-78df-4382-b728-3d67f9a80e6d)
![WhatsApp Image 2026-02-12 at 12 26 39 AM (1)](https://github.com/user-attachments/assets/4e788645-f8ba-4911-973d-f5b5e3f8eb74)

![WhatsApp Image 2026-02-12 at 12 26 39 AM (2)](https://github.com/user-attachments/assets/52cfd02b-48a6-4af7-991c-323091aa98ad)

![WhatsApp Image 2026-02-12 at 12 26 40 AM](https://github.com/user-attachments/assets/81cd5d2e-650a-4d2e-bfec-bb6cdd88e197)




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
