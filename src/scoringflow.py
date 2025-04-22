from metaflow import FlowSpec, step, Parameter, JSONType
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import os

class PredictFlow(FlowSpec):

    vector = Parameter(
        'vector',
        help="A JSON list of feature values, e.g. '[5.1,3.5,1.4,0.2]'",
        type=JSONType,
        required=True
    )

    @step
    def start(self):
        self.features = np.array(self.vector).reshape(1, -1)
        print("Received input vector:", self.features)
        self.next(self.load_model)

    @step
    def load_model(self):
        tracking_dir = os.path.abspath("mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        client = MlflowClient()

        model_name = "best_model"     
        versions = client.get_latest_versions(model_name, stages=[])
        latest = max(versions, key=lambda v: int(v.version))

        # load via the registered‐model URI
        uri = f"models:/{model_name}/{latest.version}"
        self.model = mlflow.sklearn.load_model(uri)
        print(f"Loaded '{model_name}' v{latest.version}")
        self.next(self.predict)

    @step
    def predict(self):
        self.prediction = int(self.model.predict(self.features)[0])
        print(f"Predicted class: {self.prediction}")
        self.next(self.end)

    @step
    def end(self):
        print("PredictFlow complete.")

if __name__ == '__main__':
    PredictFlow()