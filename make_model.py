import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib

headers = fetch_california_housing().feature_names
X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X, columns=headers)
X = X[['HouseAge', 'AveRooms', 'Population']]
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'california_model.sav')
