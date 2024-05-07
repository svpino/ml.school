import os
from pathlib import Path

import mlflow
from common import (
    build_features_transformer,
    build_model,
    build_target_transformer,
    load_data_from_file,
    load_data_from_s3,
)
from inference import Model
from metaflow import FlowSpec, Parameter, pypi, pypi_base, step


@pypi_base(
    python="3.10.14",
    packages={
        "python-dotenv": "1.0.1",
        "scikit-learn": "1.4.2",
        "pandas": "2.2.2",
        "numpy": "1.26.4",
        "keras": "3.3.0",
        "jax[cpu]": "0.4.26",
        "packaging": "24.0",
        "mlflow": "2.12.1",
    },
)
class TrainingFlow(FlowSpec):
    debug = Parameter(
        "debug",
        help="Whether we are debugging the flow in a local environment",
        default=False,
    )

    dataset_location = Parameter(
        "dataset_location",
        help="Location to the initial dataset",
        default="metaflow/data/",
    )

    @step
    def start(self):
        from metaflow import current

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.next(self.load_data)

    @pypi(packages={"boto3": "1.34.70"})
    @step
    def load_data(self):
        """Load the dataset in memory.

        This function reads every CSV file available and
        concatenates them into a single dataframe.
        """
        if self.debug:
            df = load_data_from_file()
        else:
            location = f"s3://{os.environ['BUCKET']}/{self.dataset_location}"

            df = load_data_from_s3(location)

        # Shuffle the data
        self.data = df.sample(frac=1, random_state=42)

        print(f"Loaded dataset with {len(self.data)} samples")

        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        import numpy as np

        self.target = np.array(self.data["species"]).reshape(-1, 1)
        self.features = self.data.drop("species", axis=1)

        self.next(self.transform_target, self.cross_validation)

    @step
    def transform_target(self):
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.target)

        self.next(self.transform_features)

    @step
    def transform_features(self):
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.features)

        self.next(self.train_model)

    @step
    def cross_validation(self):
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=5, shuffle=True)
        self.folds = list(enumerate(kfold.split(self.target, self.features)))

        self.next(self.transform_target_fold, foreach="folds")

    @step
    def transform_target_fold(self):
        """Apply the transformation pipeline to the target feature."""
        self.fold, (self.train_indices, self.test_indices) = self.input

        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            self.target[self.train_indices],
        )

        self.y_test = target_transformer.transform(
            self.target[self.test_indices],
        )

        self.next(self.transform_features_fold)

    @step
    def transform_features_fold(self):
        """Apply the transformation pipeline to the dataset features."""
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.features.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.features.iloc[self.test_indices],
        )

        self.next(self.train_model_fold)

    @step
    def train_model_fold(self):
        """Train a model as part of the cross-validation process."""
        print(f"Training fold {self.fold}...")

        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id

            mlflow.autolog()

            self.model = build_model(10, 0.01)

            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=50,
                batch_size=32,
                verbose=0,
            )

        self.next(self.evaluate_model_fold)

    @step
    def evaluate_model_fold(self):
        """Evaluate a model created as part of the cross-validation process."""
        print(f"Evaluating fold {self.fold}...")

        self.loss, self.accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2,
        )

        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                },
            )

        print(f"Fold {self.fold} - loss: {self.loss} - accuracy: {self.accuracy}")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self, inputs):
        import numpy as np

        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        metrics = [[i.accuracy, i.loss] for i in inputs]

        accuracy, loss = np.mean(metrics, axis=0)
        accuracy_std, loss_std = np.std(metrics, axis=0)

        print(f"Accuracy: {accuracy} ±{accuracy_std}")
        print(f"Loss: {loss} ±{loss_std}")

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": accuracy,
                    "cross_validation_accuracy_std": accuracy_std,
                    "cross_validation_loss": loss,
                    "cross_validation_loss_std": loss_std,
                },
            )

        self.next(self.train_model)

    @step
    def train_model(self, inputs):
        """Train the final model that will be deployed to production.

        This function will use the entire dataset to train the model.
        """
        self.merge_artifacts(inputs)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog()

            self.model = build_model(10, 0.01)

            params = {
                "epochs": 50,
                "batch_size": 32,
            }

            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                **params,
            )

            mlflow.log_params(params)

        self.next(self.register_model)

    @step
    def register_model(self):
        import tempfile

        import joblib
        from mlflow.models import ModelSignature
        from mlflow.types.schema import ColSpec, Schema

        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            tempfile.TemporaryDirectory() as directory,
        ):
            model_path = (Path(directory) / "model.keras").as_posix()
            self.model.save(model_path)

            features_transformer_path = (Path(directory) / "features.joblib").as_posix()
            target_transformer_path = (Path(directory) / "target.joblib").as_posix()
            joblib.dump(self.features_transformer, features_transformer_path)
            joblib.dump(self.target_transformer, target_transformer_path)

            input_schema = Schema(
                [
                    ColSpec("string", "island"),
                    ColSpec("double", "culmen_length_mm"),
                    ColSpec("double", "culmen_depth_mm"),
                    ColSpec("double", "flipper_length_mm"),
                    ColSpec("double", "body_mass_g"),
                ],
            )
            output_schema = Schema(
                [
                    ColSpec(type="string", name="prediction"),
                    ColSpec(type="double", name="confidence"),
                ],
            )

            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=Model(),
                artifacts={
                    "model": model_path,
                    "features_transformer": features_transformer_path,
                    "target_transformer": target_transformer_path,
                },
                pip_requirements=[
                    "pandas==2.2.2",
                    "numpy==1.26.4",
                    "keras==3.3.0",
                    "jax[cpu]==0.4.26",
                    "packaging==24.0",
                    "scikit-learn==1.4.2",
                ],
                signature=signature,
                code_paths=["inference.py"],
                registered_model_name="penguins",
            )

        self.next(self.end)

    @step
    def end(self):
        print("the end")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TrainingFlow()
