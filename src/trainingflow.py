from metaflow import FlowSpec, step, Parameter
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score
import pandas as pd

class TrainingFlow(FlowSpec):
    data_path = Parameter('data_path', help='Path to CSV training data', default='data/iris.csv')
    n_iter = Parameter('n_iter', help='Number of hyperparam iterations', default=20)

    @step
    def start(self):
        # Load raw data
        df = pd.read_csv(self.data_path)
        self.X = df.drop(columns=['target'])
        self.y = df['target']
        self.next(self.split)

    @step
    def split(self):
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.next(self.train)

    @step
    def train(self):
        # Configure MLflow
        tracking_dir = os.path.abspath('mlruns')
        mlflow.set_tracking_uri(f'file://{tracking_dir}')
        mlflow.set_experiment('TrainingFlow')

        # Hyperparameter sampling
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None]
        }
        param_list = list(ParameterSampler(param_dist, n_iter=self.n_iter, random_state=1))

        best_score = -1
        best_params = None
        best_run = None

        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f'run_{i}') as run:
                clf = RandomForestClassifier(random_state=42, **params)
                clf.fit(self.X_train, self.y_train)
                preds = clf.predict(self.X_test)
                acc = accuracy_score(self.y_test, preds)

                mlflow.log_params(params)
                mlflow.log_metric('accuracy', acc)
                mlflow.sklearn.log_model(clf, artifact_path='model')

                if acc > best_score:
                    best_score = acc
                    best_params = params
                    best_run = run.info.run_id

        self.best_run = best_run
        self.best_score = best_score
        self.next(self.register)

    @step
    def register(self):
        # Register best model
        tracking_dir = os.path.abspath('mlruns')
        mlflow.set_tracking_uri(f'file://{tracking_dir}')
        client = MlflowClient()
        model_uri = f'runs:/{self.best_run}/model'
        client.create_registered_model('best_model')
        client.create_model_version(name='best_model', source=model_uri, run_id=self.best_run)

        print(f'Registered best_model version from run {self.best_run} with accuracy {self.best_score:.4f}')
        self.next(self.end)

    @step
    def end(self):
        print('TrainingFlow complete')

if __name__ == '__main__':
    TrainingFlow()