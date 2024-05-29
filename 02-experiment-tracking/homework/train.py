import os
import pickle

import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
def run_train(data_path: str):
    print(mlflow.set_tracking_uri("sqlite:///mlflow.db"))
    print(mlflow.set_experiment("nyc-taxi-experiment"))

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    # Enable auto logging
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        mlflow.set_tag("developer", "Santiago")
        mlflow.set_tag("model", "RandomForestRegressor")

        mlflow.log_param("data_path", data_path)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 0)
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact(os.path.join(data_path, "train.pkl"))
        print(rmse)


if __name__ == "__main__":
    run_train()
