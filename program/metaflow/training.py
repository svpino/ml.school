import os
from pathlib import Path

from common import (
    PACKAGES,
    build_features_transformer,
    build_model,
    build_target_transformer,
    load_dataset,
)
from inference import Model
from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    project,
    pypi,
    pypi_base,
    retry,
    step,
)
from metaflow.cards import Artifact, Markdown, ProgressBar, Table


@project(name="penguins")
@pypi_base(
    python="3.10.14",
    packages=PACKAGES,
)
class TrainingFlow(FlowSpec):
    dataset = IncludeFile(
        "penguins",
        is_text=True,
        help="Penguins dataset",
        default="../penguins.csv",
    )

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy threshold to register the model",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start and set up the pipeline."""
        import mlflow

        mode = "production" if current.is_production else "development"
        print(f"Running flow in {mode} mode.")

        # We want to use each specific Metaflow run to identify the experiment in
        # MLFlow. We can use the `run_id` attribute to accomplish this.
        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        current.card.append(Markdown("## Configuration"))
        current.card.append(
            Table(
                [
                    [Markdown("**Mode**"), Artifact(mode)],
                    [Markdown("**MLFlow Experiment**"), Artifact(current.run_id)],
                ],
            ),
        )

        self.next(self.load_data)

    @pypi(packages={"boto3": "1.34.70"})
    @retry
    @card
    @step
    def load_data(self):
        """Load the dataset in memory."""
        dataset = os.environ["DATASET"] if current.is_production else self.dataset
        self.data = load_dataset(
            dataset,
            is_production=current.is_production,
        )

        print(f"Loaded dataset with {len(self.data)} samples")

        self.next(self.cross_validation, self.transform)

    @step
    def cross_validation(self):
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=5, shuffle=True)
        self.folds = list(enumerate(kfold.split(self.data)))

        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Apply the transformation pipeline to the target feature."""
        self.fold, (self.train_indices, self.test_indices) = self.input
        species = self.data.species.to_numpy().reshape(-1, 1)

        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            species[self.train_indices],
        )
        self.y_test = target_transformer.transform(
            species[self.test_indices],
        )

        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.data.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.data.iloc[self.test_indices],
        )

        self.next(self.train_model_fold)

    @step
    def train_model_fold(self):
        """Train a model as part of the cross-validation process."""
        import mlflow

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

            self.model = build_model()
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
        import mlflow

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

    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the cross-validation process.

        This function averages the metrics computed for each individual fold to
        determine the final model performance.
        """
        import mlflow
        import numpy as np

        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        metrics = [[i.accuracy, i.loss] for i in inputs]

        self.accuracy, loss = np.mean(metrics, axis=0)
        accuracy_std, loss_std = np.std(metrics, axis=0)

        print(f"Accuracy: {self.accuracy} ±{accuracy_std}")
        print(f"Loss: {loss} ±{loss_std}")

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.accuracy,
                    "cross_validation_accuracy_std": accuracy_std,
                    "cross_validation_loss": loss,
                    "cross_validation_loss_std": loss_std,
                },
            )

        self.next(self.train_model)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the dataset.

        This function transforms the columns of the entire dataset because we'll
        use all of the data to train the final model.
        """
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(
            self.data.species.to_numpy().reshape(-1, 1),
        )

        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        self.next(self.train_model)

    @card(refresh_interval=1)
    @step
    def train_model(self, inputs):
        """Train the final model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import mlflow
        from keras.callbacks import LambdaCallback

        self.merge_artifacts(inputs)

        p = ProgressBar(max=50, label="Epochs")
        current.card.append(p)
        current.card.refresh()

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog()

            self.model = build_model()

            params = {
                "epochs": 50,
                "batch_size": 32,
            }

            def refresh_progress(epoch, logs):
                p.update(epoch + 1)
                current.card.refresh()

            progress_callback = LambdaCallback(on_epoch_end=refresh_progress)

            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                callbacks=[progress_callback],
                **params,
            )

            mlflow.log_params(params)

        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model in the MLFlow Model Registry."""
        import tempfile

        import joblib
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types.schema import ColSpec, Schema

        if self.accuracy >= self.accuracy_threshold:
            print("Registering model...")

            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                model_path = (Path(directory) / "model.keras").as_posix()
                self.model.save(model_path)

                features_transformer_path = (
                    Path(directory) / "features.joblib"
                ).as_posix()
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
                        ColSpec("string", "sex"),
                    ],
                )
                output_schema = Schema(
                    [
                        ColSpec(type="string", name="prediction"),
                        ColSpec(type="double", name="confidence"),
                    ],
                )

                signature = ModelSignature(
                    inputs=input_schema,
                    outputs=output_schema,
                )

                input_example = [
                    {
                        "island": "Biscoe",
                        "culmen_length_mm": 48.6,
                        "culmen_depth_mm": 16.0,
                        "flipper_length_mm": 230.0,
                        "body_mass_g": 5800.0,
                        "sex": "MALE",
                    },
                    {
                        "island": "Torgersen",
                        "culmen_length_mm": 44.1,
                        "culmen_depth_mm": 18.0,
                        "flipper_length_mm": 210.0,
                        "body_mass_g": 4000,
                    },
                ]

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
                        "keras==3.3.3",
                        "jax[cpu]==0.4.28",
                        "packaging==24.0",
                        "scikit-learn==1.5.0",
                    ],
                    signature=signature,
                    code_paths=["inference.py"],
                    registered_model_name="penguins",
                    input_example=input_example,
                )
        else:
            print(
                f"The accuracy of the model ({self.accuracy:.2f}) is lower than the "
                f"accuracy threshold ({self.accuracy_threshold}). "
                "Skipping model registration.",
            )

        self.next(self.end)

    @step
    def end(self):
        print("the end")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TrainingFlow()
