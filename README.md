# Implementation-of-Linear-Regression-Using-Gradient-Descent
# TAMIZHAN B
# 212225230283
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.

2.Write a function compute Cost to generate the cost function.

3.Perform iterations of gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: TAMIZHAN B
RegisterNumber:  212225230283
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

A = data['R&D Spend'].values
b = data['Profit'].values

A = (A - A.mean()) / A.std()

c = 0
d = 0

learning_rate = 0.01
epochs = 1000
n = len(A)

for i in range(epochs):
    b_pred = c * A + d
    
    dm = (-2/n) * np.sum(A * (b - b_pred))
    db = (-2/n) * np.sum(b - b_pred)
    
    c = c - learning_rate * dm
    d = d - learning_rate * db

print("Slope (c):", c)
print("Intercept (d):", d)

b_pred = c * A + d

plt.scatter(A, b)
plt.plot(A, b_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()

```

## Output:
<img width="1092" height="770" alt="image" src="https://github.com/user-attachments/assets/ed0f102a-0c16-413f-9e7c-e62fed002d85" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
