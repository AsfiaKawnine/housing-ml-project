import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

from data_preprocessing import create_preprocessor

df = pd.read_csv("../data/house-prices-advanced-regression-techniques/train.csv")


X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

preprocessor = create_preprocessor(X_train)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor())
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "../models/housing_model_2.pkl")