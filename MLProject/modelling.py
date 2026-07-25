import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("SMSML_Experiment")

train = pd.read_csv("train_preprocessed.csv")

features = [col for col in train.columns if col != "target"]
x_train, x_test, y_train, y_test = train_test_split(
    train[features],
    train["target"],
    test_size=0.2,
    random_state=42,
    stratify=train["target"],
)

input_example = x_train.head(3)

with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
    mlflow.sklearn.autolog()

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
    )
