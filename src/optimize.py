# src/optimize.py
import optuna
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def objective(trial):
    # load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # load data
    data = pd.read_csv("data_clean.csv")
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"])

    # Choose model
    model_type = trial.suggest_categorical("model_type", ["random_forest", "linear_regression", "gradient_boosting"])
    
    if model_type == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    elif model_type == "linear_regression":
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        model = LinearRegression(fit_intercept=fit_intercept)

    elif model_type == "gradient_boosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, predictions))
    return rmse

# Run optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  RMSE: {trial.value}")
    print(f"  Params: {trial.params}")

    # save params
    with open("best_params.yaml", "w") as f:
        yaml.dump(trial.params, f)
