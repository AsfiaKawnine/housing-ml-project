import joblib
import pandas as pd

model = joblib.load("../models/housing_model_2.pkl")

def predict(input_dict):
    # Create full input with ALL required columns
    full_input = {col: 0 for col in model.feature_names_in_}

    # Update with user input
    full_input.update(input_dict)

    df = pd.DataFrame([full_input])

    prediction = model.predict(df)[0]
    return float(prediction)

sample_input = {
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "TotalBsmtSF": 800,
    "FullBath": 2,
    "YearBuilt": 2005
}

result = predict(sample_input)
print("Prediction:", result)