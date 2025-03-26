import os
from pathlib import Path

from common import (
    PYTHON,
    DatasetMixin,
    Pipeline,
    build_features_transformer,
    build_model,
    build_target_transformer,
    packages,
)
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    current,
    environment,
    project,
    step,
)


@project(name="penguins")
@conda_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "tensorflow",
        "boto3",
        "mlflow",
    ),
)
class Training(FlowSpec, Pipeline, DatasetMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="Location of the MLflow tracking server.",
        default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"),
    )

    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs that will be used to train the model.",
        default=50,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=32,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        logger = self.configure_logging()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logger.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logger.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset(logger)

        try:
            # Let's start a new MLflow run to track the execution of this flow. We want
            # to set the name of the MLflow run to the Metaflow run ID so we can easily
            # recognize how they relate to each other.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process. We'll shuffle the data
        # before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We can use a `foreach` to run every fold on a separate branch. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold.
        self.fold, (self.train_indices, self.test_indices) = self.input
        logger = self.configure_logging()
        logger.info("Transforming fold %d...", self.fold)

        # We can use the indices to split the data into training and test sets.
        train_data = self.data.iloc[self.train_indices]
        test_data = self.data.iloc[self.test_indices]

        # Let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(train_data)
        self.x_test = features_transformer.transform(test_data)

        # Finally, we can build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(train_data)
        self.y_test = target_transformer.transform(test_data)

        # After processing the data and storing it as artifacts in the flow, we can move
        # to the training step.
        self.next(self.train_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
        },
    )
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logger = self.configure_logging()
        logger.info("Training fold %d...", self.fold)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # We want to track the training process under the same MLflow run we started at
        # the beginning of the flow. Since we are running cross-validation, we will
        # create a nested run for each fold to keep track of each model individually.
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model.
            self.mlflow_fold_run_id = run.info.run_id

            # We are currently training a model corresponding to an individual fold,
            # so we don't want to log that model because it's useless.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the training data we processed in the
            # previous step.
            self.model = build_model(self.x_train.shape[1])
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                verbose=0,
            )

        logger.info(
            "Fold %d - train_loss: %f - train_accuracy: %f",
            self.fold,
            history.history["loss"][-1],
            history.history["accuracy"][-1],
        )

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
        },
    )
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data associated with the current fold.
        """
        import mlflow

        logger = self.configure_logging()
        logger.info("Evaluating fold %d...", self.fold)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Let's evaluate the model using the test data we processed before.
        self.test_loss, self.test_accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0,
        )

        logger.info(
            "Fold %d - test_loss: %f - test_accuracy: %f",
            self.fold,
            self.test_loss,
            self.test_accuracy,
        )

        # Let's track the evaluation metrics under the nested MLflow run corresponding
        # to the current fold.
        mlflow.log_metrics(
            {
                "test_loss": self.test_loss,
                "test_accuracy": self.test_accuracy,
            },
            run_id=self.mlflow_fold_run_id,
        )

        # When we finish evaluating the models in the cross-validation process, we want
        # to average the scores to determine the overall model performance.
        self.next(self.average_scores)

    @card
    @step
    def average_scores(self, inputs):
        """Averages the scores computed for each individual model."""
        import mlflow
        import numpy as np

        logger = self.configure_logging()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # We need access to the `mlflow_run_id` artifact that we set at the start of
        # the flow, but since we are in a join step, we need to merge the artifacts
        # from the incoming branches to make `mlflow_run_id` available.
        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds.
        metrics = [[i.test_accuracy, i.test_loss] for i in inputs]
        self.test_accuracy, self.test_loss = np.mean(metrics, axis=0)
        self.test_accuracy_std, self.test_loss_std = np.std(metrics, axis=0)

        logger.info("Accuracy: %f ±%f", self.test_accuracy,
                    self.test_accuracy_std)
        logger.info("Loss: %f ±%f", self.test_loss, self.test_loss_std)

        # Let's log the model metrics on the parent run.
        mlflow.log_metrics(
            {
                "test_accuracy": self.test_accuracy,
                "test_accuracy_std": self.test_accuracy_std,
                "test_loss": self.test_loss,
                "test_loss_std": self.test_loss_std,
            },
            run_id=self.mlflow_run_id,
        )

        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register the final version of the model.
        self.next(self.register)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        We'll use the entire dataset to build the final model, so we need to transform
        the dataset before training.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline and transform the dataset features.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Let's build the SciKit-Learn pipeline and transform the target column.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
        },
    )
    @step
    def train(self):
        """Train the final model using the entire dataset."""
        import mlflow

        logger = self.configure_logging()
        logger.info("Training final model...")

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Let's log the training process under the current MLflow run.
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # We want to log the model manually, so let's disable automatic logging.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the entire dataset.
            self.model = build_model(self.x.shape[1])
            self.model.fit(
                self.x,
                self.y,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                verbose=2,
            )

        # After we finish training the model, we want to register it.
        self.next(self.register)

    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
        },
    )
    @step
    def register(self, inputs):
        """Register the model in the model registry.

        This function will prepare and register the final model in the model registry
        if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        logger = self.configure_logging()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the
        # `accuracy_threshold` parameter.
        if self.test_accuracy >= self.accuracy_threshold:
            self.registered = True
            logger.info("Registering model...")

            # We'll register the model under the current MLflow run. We also need to
            # create a temporary directory to store the model artifacts.
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                root = Path(__file__).parent
                self.code_paths = [
                    (root / "inference" / "backend.py").as_posix()]

                # We can now register the model in the model registry. This will
                # automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    python_model=Path(__file__).parent /
                    "inference" / "model.py",
                    registered_model_name="penguins",
                    artifact_path="model",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements,
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )
        else:
            self.registered = False
            logger.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logger = self.configure_logging()
        logger.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (
            Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (
            Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}" if version else package
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "keras",
                "tensorflow",
            ).items()
        ]


if __name__ == "__main__":
    Training()
