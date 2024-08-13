"""Something."""

import os
from pathlib import Path

from common import (
    PACKAGES,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
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
    """Flow implementing the training pipeline.

    This flow trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    dataset = IncludeFile(
        "penguins",
        is_text=True,
        help=(
            "Local copy of the penguins dataset. This file will be included in the "
            "flow and will be used whenever the flow is executed in development mode."
        ),
        default="../penguins.csv",
    )

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
        ),
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start and prepare the Training flow."""
        import mlflow

        mode = "production" if current.is_production else "development"
        print(f"Running flow in {mode} mode.")

        # Let's start a new MLFlow run to track everything that happens during the
        # execution of this flow. We want to set the name of the MLFlow experiment to
        # the Metaflow run identifier so we can easily recognize which experiment
        # corresponds with each run.
        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        # This is the configuration we'll use to train the model. We want to set it up
        # at this point so we can reuse it later throughout the flow.
        self.training_parameters = {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }

        self._populate_card_start_step(mode)

        # Now that everything is set up, let's load the dataset.
        self.next(self.load_data)

    @pypi(packages={"boto3": "1.34.70"})
    @retry
    @card
    @step
    def load_data(self):
        """Load the dataset in memory."""
        # TODO: Exception if the env var is not set.
        dataset = os.environ["DATASET"] if current.is_production else self.dataset

        # Load the dataset in memory. This function will either read the dataset from
        # the included file or from an S3 location, depending on the mode in which the
        # flow is running.
        self.data = load_dataset(
            dataset,
            is_production=current.is_production,
        )

        print(f"Loaded dataset with {len(self.data)} samples")

        # Now that we loaded the data, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process to evaluate the model,
        # so let's set it up. We'll shuffle the data before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these values in the previous step and passed
        # them as the input to this step.
        self.fold, (self.train_indices, self.test_indices) = self.input

        print(f"Transforming fold {self.fold}...")

        # We need to turn the target column into a shape that the Scikit-Learn
        # pipeline understands.
        species = self.data.species.to_numpy().reshape(-1, 1)

        # We can now build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            species[self.train_indices],
        )
        self.y_test = target_transformer.transform(
            species[self.test_indices],
        )

        # Finally, let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.data.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.data.iloc[self.test_indices],
        )

        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        self.next(self.train_model_fold)

    @step
    def train_model_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        print(f"Training fold {self.fold}...")

        # TODO: Explain this
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # TODO: Explain this
            self.mlflow_fold_run_id = run.info.run_id

            # TODO: Explain this
            mlflow.autolog()

            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `transform` step.
            self.model = build_model()
            self.model.fit(
                self.x_train,
                self.y_train,
                verbose=0,
                **self.training_parameters,
            )

        # After training the model, we want to evaluate it using the test data.
        self.next(self.evaluate_model_fold)

    @card
    @step
    def evaluate_model_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow

        print(f"Evaluating fold {self.fold}...")

        # Let's evaluate the model using the test data we processed and stored as
        # artifacts during the `transform` step.
        self.loss, self.accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2,
        )

        # TODO: Add these values to the card
        print(f"Fold {self.fold} - loss: {self.loss} - accuracy: {self.accuracy}")

        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                },
            )

        # TODO: Add comment here
        self.next(self.evaluate_model)

    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        import mlflow
        import numpy as np

        # TODO: Explain how this works
        self.merge_artifacts(inputs, include=["mlflow_run_id"])

        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds. Notice how we are accumulating these values
        # using the `inputs` parameter provided by Metaflow.
        metrics = [[i.accuracy, i.loss] for i in inputs]
        self.accuracy, loss = np.mean(metrics, axis=0)
        accuracy_std, loss_std = np.std(metrics, axis=0)

        # TODO: Add these values to the card
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

        # TODO: Add comment here
        self.next(self.train_model)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        This function transforms the columns of the entire dataset because we'll
        use all of the data to train the final model.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline to process the target column and use it
        # to transform the data.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(
            self.data.species.to_numpy().reshape(-1, 1),
        )

        # Let's build the SciKit-Learn pipeline to process the feature columns and use
        # it to transform the training.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train_model)

    @card(refresh_interval=1)
    @step
    def train_model(self, inputs):
        """Train the final model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import mlflow
        from keras.callbacks import LambdaCallback

        # TODO: Add comment here
        self.merge_artifacts(inputs)

        # We want to display a progress bar in the Metaflow card that
        # shows the progress of the training process.
        p = ProgressBar(max=TRAINING_EPOCHS, label="Epochs")
        current.card.append(p)
        current.card.refresh()

        # TODO: Explain
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog()

            # TODO: Comment
            self.model = build_model()

            # TODO: Comment
            def on_epoch_end(epoch, logs):
                p.update(epoch + 1)
                current.card.refresh()

            # TODO: Comment
            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)],
                **self.training_parameters,
            )

            # TODO: Comment
            mlflow.log_params(self.training_parameters)

        # TODO: Comment
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model in MLFlow's Model Registry."""
        import tempfile

        import mlflow

        # We only want to register the model if its accuracy is above the threshold
        # specified by the flow's parameter.
        if self.accuracy >= self.accuracy_threshold:
            print("Registering model...")

            # TODO: Comment
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # TODO: Comment
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    code_paths=["inference.py"],
                    registered_model_name="penguins",
                    python_model=Model(),
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    input_example=self._get_model_input_example(),
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )
        else:
            print(
                f"The accuracy of the model ({self.accuracy:.2f}) is lower than the "
                f"accuracy threshold ({self.accuracy_threshold}). "
                "Skipping model registration.",
            )

        # TODO: Comment
        self.next(self.end)

    @step
    def end(self):
        # TODO: Do I need this?
        print("the end")

    def _populate_card_start_step(self, mode: str):
        # TODO: Add documentation to this function
        current.card.append(Markdown("## Configuration"))
        current.card.append(
            Table(
                [
                    [Markdown("**Mode**"), Artifact(mode)],
                    [Markdown("**MLFlow Experiment**"), Artifact(current.run_id)],
                ],
            ),
        )

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
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input=self._get_model_input_example(),
            model_output={"prediction": "Adelie", "confidence": 0.90},
        )

    def _get_model_input_example(self):
        """Return an input example for the model.

        Including an input example when logging a model helps in inferring the model's
        signature and validates the model's requirements.
        """
        return {
            "island": "Biscoe",
            "culmen_length_mm": 48.6,
            "culmen_depth_mm": 16.0,
            "flipper_length_mm": 230.0,
            "body_mass_g": 5800.0,
            "sex": "MALE",
        }

    def _get_model_pip_requirements(self):
        """Return the list of required libraries to run the model in production.

        This function uses the `PACKAGES` dictionary to determine the proper version of
        the libraries that must be installed during inference time.
        """
        requirements = [
            "pandas",
            "numpy",
            "keras",
            "jax[cpu]",
            "packaging",
            "scikit-learn",
        ]
        return [f"{package}=={PACKAGES[package]}" for package in requirements]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TrainingFlow()
