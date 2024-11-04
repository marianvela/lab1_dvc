# src/train.py
import pandas as pd
import joblib
import sys
import yaml
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def train(input_file, model_file, params_file):
    # clean dataset
    df = pd.read_csv(input_file)

    # load yaml parameters
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # feature and target split
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    X = df[features]
    y = df[target]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    # model selection loop
    model_type = params["model"]["type"]

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params["model"]["random_forest"]["n_estimators"],
            max_depth=params["model"]["random_forest"]["max_depth"]
        )
    elif model_type == "linear_regression":
        model = LinearRegression(
            fit_intercept=params["model"]["linear_regression"]["fit_intercept"]
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=params["model"]["gradient_boosting"]["n_estimators"],
            learning_rate=params["model"]["gradient_boosting"]["learning_rate"],
            max_depth=params["model"]["gradient_boosting"]["max_depth"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # train
    model.fit(X_train, y_train)

    # save the model
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}")

if __name__ == "__main__":
    # args: input , model , params
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)