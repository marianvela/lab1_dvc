# src/evaluate.py
import joblib
import pandas as pd
import sys
import yaml
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def evaluate(model_file, data_file, params_file):
    # load model
    model = joblib.load(model_file)

    # load test data
    data = pd.read_csv(data_file)

    # load params
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # split target and independent variables
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    X = data[features]
    y = data[target]

    # predict using the test split
    predictions = model.predict(X)

    # calculate RMSE and MAE performance metrics
    rmse = sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    # show results
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")

    # save results to a JSON file
    with open("metrics.json", "w") as f:
         json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=4)

if __name__ == "__main__":
    # args: model , test data , params 
    model_file = sys.argv[1]
    data_file = sys.argv[2]
    params_file = sys.argv[3]

    evaluate(model_file, data_file, params_file)
