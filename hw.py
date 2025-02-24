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
YtestPredict = linModel.predict(Xtest)

rmseLinModel = (np.sqrt(mean_squared_error(Ytest, YtestPredict)))
print("rmse in case of linear regression: ", rmseLinModel)

from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree=2)

XtrainPoly = polyFeature.fit_transform(Xtrain)

polyModel = LinearRegression()
polyModel.fit(XtrainPoly, Ytrain)

XtestPoly = polyFeature.fit_transform(Xtest)
YtestPredictPoly = polyModel.predict(XtestPoly)

rmsePolyModel = (np.sqrt(mean_squared_error(Ytest, YtestPredictPoly)))
print("rmse in case of polynomial regression: ", rmsePolyModel)