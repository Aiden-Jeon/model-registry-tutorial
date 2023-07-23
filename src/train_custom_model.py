import os

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

mlflow.set_experiment("tutorial")


class MyModel:
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        import pandas as pd

        X_pred = self.clf.predict(X)
        X_pred_df = pd.Series(X_pred).map({0: "virginica", 1: "setosa", 2: "versicolor"})
        return X_pred_df


#
# load data
#
iris = load_iris(as_frame=True)
X, y = iris["data"], iris["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2024)


with mlflow.start_run():
    parameters = {
        "n_estimators": 100,
        "max_depth": 5,
    }
    mlflow.log_params(parameters)
    #
    # train model
    #
    clf = RandomForestClassifier(
        n_estimators=parameters["n_estimators"], max_depth=parameters["max_depth"], random_state=2024
    )
    clf.fit(X_train, y_train)

    #
    # evaluate train model
    #
    y_pred = clf.predict(X_valid)
    acc_score = accuracy_score(y_valid, y_pred)

    print("Accuracy score is {:.4f}".format(acc_score))
    mlflow.log_metric("accuracy", acc_score)

    #
    # save model
    #
    my_model = MyModel(clf=clf)
    mlflow.sklearn.log_model(sk_model=my_model, artifact_path="my_model")
