# api/app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

model = joblib.load("E:/housing-ml-project/models/housing_model.pkl")

app = FastAPI()

class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int


@app.post("/predict")
def predict(data: HouseInput):
    input_dict = data.dict()

    # Create full input with ALL columns
    full_input = {col: 0 for col in model.feature_names_in_}

    # Update with actual values
    full_input.update(input_dict)

    df = pd.DataFrame([full_input])

    prediction = model.predict(df)[0]

    return {"predicted_price": float(prediction)}