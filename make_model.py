import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import joblib

headers = load_boston().feature_names[:3]
# print(load_boston().DESCR)
X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X[:, 0:3], columns=headers)
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'boston_model.sav')
