import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)

print(housing.keys())

ames = pd.DataFrame(housing.data, columns=housing.feature_names)
print(ames.head())

ames["MEDV"] = housing.target

X = ames[["LSTAT", "RM"]]
Y = ames["MEDV"]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=5)

from sklearn.linear_model import LinearRegression
linModel = LinearRegression()
linModel.fit(Xtrain, Ytrain)

from sklearn.metrics import mean_squared_error
yTestPredict = linModel.predict(Xtest)

rmseLinModel = (np.sqrt(mean_squared_error(Ytest, yTestPredict)))
print("rmse in case of linear regression: ", rmseLinModel)