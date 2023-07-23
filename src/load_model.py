import os

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"


def load_sklearn_model(run_id, model_name):
    clf = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
    return clf


def load_pyfunc_model(run_id, model_name):
    clf = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")
    return clf


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--model-name", type=str, default="my_model")
    args = parser.parse_args()

    #
    # load data
    #
    iris = load_iris()
    X, y = iris["data"], iris["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2024)

    #
    # load model
    #
    sklearn_clf = load_sklearn_model(args.run_id, args.model_name)
    sklearn_pred = sklearn_clf.predict(X)
    print("sklearn")
    print(sklearn_clf)
    print(sklearn_pred)

    pyfunc_clf = load_pyfunc_model(args.run_id, args.model_name)
    pyfunc_pred = pyfunc_clf.predict(X)
    print("pyfunc")
    print(pyfunc_clf)
    print(pyfunc_pred)
