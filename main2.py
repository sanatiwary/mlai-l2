import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data
y = boston.target

print(X.head())

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=5)

from sklearn.linear_model import LinearRegression
linModel = LinearRegression()
linModel.fit(Xtrain, Ytrain)

from sklearn.metrics import mean_squared_error
yTestPredict = linModel.predict(Xtest)

rmseLinModel = (np.sqrt(mean_squared_error(Ytest, yTestPredict)))
print("rmse in case of linear regression: ", rmseLinModel)