#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\adamg\Documents\ML_Research_Final_CSV.csv", header=None) #file contains no header info
print(f"Read in {len(df)} rows")
df1 = pd.read_csv(r"C:\Users\adamg\Documents\ML_Research_Final_CSV.csv")#, header=None)
df.head()

df.replace("?", 10000, inplace=True) 

fig1=sns.pairplot(data=df1)
plt.show()
df.drop([0], 0, inplace=True)
print('Data:',df)
df.head()



import numpy as np
from sklearn.model_selection import train_test_split

X_1 = np.array(df.drop([0], 1)) 

y_1 = np.array(df[0]) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit
start = timeit.default_timer()
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=51)

X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.4, random_state=43)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print(f"Accuracy of MLP Regressor is:{clf.score}")
print("clf = ",clf.score)
print("X_train = ",X_train.shape)
print("X_test = ",X_test.shape)
print("y_train = ",y_train.shape)
print("y_test = ",y_test.shape)

nn_model = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=43, max_iter=1000, learning_rate='adaptive')

nn_model.fit(X_train, y_train)
nn_accuracy = nn_model.score(X_test, y_test)# Why is accuracy not used here

stop = timeit.default_timer()
prediction = nn_model.predict(X_test)
prediction=prediction.astype('float')
y_test=y_test.astype('float')
mse = mean_squared_error(y_test, prediction)

print('Time:',stop-start)
print(f"Mean Squared Error is :{mse}")


nn_model.fit(X_train, y_train)           
plt.plot(nn_model.loss_curve_,label="train") 
plt.plot(nn_model.loss_curve_,label="test") 
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')






