import numpy as np
import pandas as pd
import seaborn as sns

titanic = pd.read_csv("titanic.csv")

X = titanic[["Age", "Fare"]]
Y = titanic["Survived"]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=5)

from sklearn.linear_model import LinearRegression
linModel = LinearRegression()
linModel.fit(Xtrain, Ytrain)

from sklearn.metrics import mean_squared_error
yTestPredict = linModel.predict(Xtest)

rmseLinModel = (np.sqrt(mean_squared_error(Ytest, yTestPredict)))
print("rmse in case of linear regression: ", rmseLinModel)
