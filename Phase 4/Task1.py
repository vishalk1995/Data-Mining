# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:20:03 2019

@author: rockers_vn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def myMSR(true_y, pre_y):
    mse = 0
    for i in range(len(true_y)):
        mse = mse + (true_y[i]-pre_y[i])**2
    mse = mse/len(true_y)
    return mse

colnames=['x1','x2','y'] 
test_df = pd.read_csv('PB1_test.csv', names=colnames, header=None)
train_df = pd.read_csv('PB1_train.csv', names=colnames, header=None)
X = train_df[['x1','x2']]
X = np.array(X)
Y = np.array(train_df['y'])
reg = LinearRegression().fit(X, list(Y))
coef = reg.coef_
intercept = reg.intercept_
test_X = np.array(test_df[['x1','x2']])
test_y_true = np.array(test_df['y'])
predict_y = reg.predict(test_X)


test_error = myMSR(test_y_true, predict_y)
single_y = reg.predict([46,53])


#result
print('\n1. value of thetas:\n')
print('theta 0 - '+str(intercept))
print('theta 1 - '+str(coef[0]))
print('theta 2 - '+str(coef[1]))
print('\n')
print('2. Predicted values on "PB1_test.csv"\n')
print(predict_y)

print('\n3. MSE on test set"\n')
print(test_error)
print('\n4. Y value for x =[46, 53]\n')
print(single_y[0])
print('\n5. Required graph \n see Task1.png')

#plot
x1, x2 = np.meshgrid(range(int(max(test_df.x1))), range(int(max(test_df.x2))))
y = coef[0]*x1 + coef[1]*x2 + intercept 

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(test_df.x1, test_df.x2, test_df.y, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.plot_surface(x1, x2, y, alpha=0.2)
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_zlabel('Y value')
plt.show()