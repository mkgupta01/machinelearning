#linear regression
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt 

# sklearn provide free dataset for programmers to learn ml 
#the datasets used here is diabetes

diabetes =datasets.load_diabetes()
# print(diabetes.keys())
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabetes_x=diabetes.data #getting features


diabetes_x_train=diabetes_x[:-30]#last 30 data is used for training model
diabetes_x_test=diabetes_x[-30:]#last 30 data is used for testing model

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)#training model
diabetes_y_model=model.predict(diabetes_x_test)#testing model

# plt.scatter(diabetes_x_test,diabetes_y_test)
# plt.plot(diabetes_x_test,diabetes_y_model)
# plt.show()

print(mean_squared_error(diabetes_y_test,diabetes_y_model))



