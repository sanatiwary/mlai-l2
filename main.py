import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

bostonData = load_boston()
print(bostonData.keys())

boston = pd.DataFrame(bostonData.data, columns=bostonData.feature_names)
print(boston.head())

boston["MEDV"] = bostonData.target

X = boston[["LSTAT", "RM"]]
Y = boston["MEDV"]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=5)

from sklearn.linear_model import LinearRegression
linModel = LinearRegression()
linModel.fit(Xtrain, Ytrain)

from sklearn.metrics import mean_squared_error
yTestPredict = linModel.predict(Xtest)

rmseLinModel = (np.sqrt(mean_squared_error(Ytest, yTestPredict)))
print("rmse in case of linear regression: ", rmseLinModel)