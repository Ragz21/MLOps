from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from pydantic import BaseModel

app = FastAPI(
    title="Iris Classifier",
    description="",
    version="0.1",
)

class InputVector(BaseModel):
    vector: list

@app.get('/')
def main():
	return {'message': 'This is a model for classifying Iris species from features'}

class request_body(BaseModel):
    reddit_comment : str

@app.on_event('startup')
def load_artifacts():
    global model

    tracking_dir = os.path.abspath("../mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    client = MlflowClient()

    model_name = "best_model"     
    versions = client.get_latest_versions(model_name, stages=[])
    latest = max(versions, key=lambda v: int(v.version))

    uri = f"models:/{model_name}/{latest.version}"
    model = mlflow.sklearn.load_model(uri)
    print(f"Loaded '{model_name}' v{latest.version}")


@app.post('/predict')
def predict(data : InputVector):
    class_to_species = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }
    features = np.array(data.vector).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    species_name = class_to_species.get(prediction, "Unknown")
    return {'Predicted Species': species_name}