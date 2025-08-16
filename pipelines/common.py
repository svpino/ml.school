import importlib
import os
import re
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd
from metaflow import (
    Config,
    FlowMutator,
    FlowSpec,
    Parameter,
    config_expr,
    current,
    project,
    user_step_decorator,
)

PYTHON = "3.12.8"

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.2",
    "tensorflow": "2.18.0",
    "evidently": "0.7.4",
}


@user_step_decorator
def dataset(step_name, flow, inputs=None, attr=None):  # noqa: ARG001
    """Load and prepare the dataset.

    This decorator loads the dataset, cleans the sex column by replacing extraneous
    values with NaN, drops any rows with missing values, and then shuffles the
    data before creating an artifact on the current flow.
    """
    import numpy as np

    # Let's check if the dataset file exists
    if not Path(flow.dataset).exists():
        # If we don't find the dataset file, we can set the artifact to None
        # and let the step continue.
        flow.data = None
        yield
    else:
        # If we find the dataset file, we can load it and process it.
        data = pd.read_csv(flow.dataset)

        # Replace extraneous values in the sex column with NaN
        data["sex"] = data["sex"].replace(".", np.nan)

        # Drop rows with missing values
        row_count_before = len(data)
        data = data.dropna()
        flow.logger.info(
            "Dropped %d rows with missing values", row_count_before - len(data)
        )

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        flow.logger.info("Loaded dataset with %d samples", len(data))

        # Let's now create an artifact on the current flow.
        flow.data = data
        yield


class logging(FlowMutator):  # noqa: N801
    """Add the @logger decorator to every step of a flow."""

    def mutate(self, mutable_flow):
        """Mutates the supplied flow by applying the @logger decorator to all steps."""
        for _, step in mutable_flow.steps:
            step.add_decorator("logger", duplicates=step.IGNORE)


@user_step_decorator
def logger(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Configure the logging handler and set it as an artifact on the step."""
    import logging
    import logging.config

    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

    flow.logger = logging.getLogger("mlschool")
    yield


@user_step_decorator
def backend(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Instantiate the backend class using the supplied configuration."""
    # For the configuration to remain clean and easy to remember, we want
    # to reference backend classes as "backend.<class_name>" without having
    # to include the full class path "inference.backend.<class_name>".
    # To accomplish this, we need to import the "inference.backend" module
    # here, so it's available to the `import_module` function below.
    import inference.backend  # noqa: F401

    # If the backend module was not specified as part of the configuration,
    # we'll default to the "Local" implementation.
    backend_module = flow.backend.get("backend", "backend.Local")

    try:
        module, cls = backend_module.rsplit(".", 1)
        module = importlib.import_module(module)
        backend_impl = getattr(module, cls)(config=flow.backend)
    except Exception as e:
        message = f"There was an error instantiating class {backend_module}"
        flow.logger.exception(message)
        raise RuntimeError(message) from e
    else:
        flow.logger.info("Backend: %s", backend_module)
        flow.backend_impl = backend_impl
        yield


def parse_backend_configuration(x):
    """Parse the backend configuration from the supplied TOML file.

    This function will expand any environment variables that are present in the
    configuration values. The environment variables should be in the format
    `${ENVIRONMENT_VARIABLE}`.
    """
    config = tomllib.loads(x).get("backend", {})

    # This regex matches any environment variable in the format ${ENVIRONMENT_VARIABLE}
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"${{{env_var}}}")

    for key, value in config.items():
        if isinstance(value, str):
            config[key] = pattern.sub(replacer, value)

    return config


@logging
@project(name=config_expr("project.project"))
class Pipeline(FlowSpec):
    """Foundation flow for pipelines that require access to the dataset and backend."""

    project = Config(
        "project",
        help="Project configuration settings.",
        default="config/local.toml",
        parser=tomllib.loads,
    )

    backend = Config(
        "backend",
        help="Backend configuration settings.",
        default="config/local.toml",
        parser=parse_backend_configuration,
    )

    dataset = Parameter(
        "dataset",
        help="Project dataset that will be used to train and evaluate the model.",
        default="data/penguins.csv",
    )


def packages(*names: str):
    """Return a dictionary of the specified packages and their corresponding version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.

    Any packages that should be locked to a specific version will be part of the
    `PACKAGES` dictionary. If a package is not present in the dictionary, it will be
    installed using the latest version available.
    """
    return {name: PACKAGES.get(name, "") for name in names}


def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # We can use the `handle_unknown="ignore"` parameter to ignore unseen categories
        # during inference. When encoding an unknown category, the transformer will
        # return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                # We'll apply the numeric transformer to all columns that are not
                # categorical (object).
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which is also a
                # categorical column. To accomplish this, we can specify the column
                # names we only want to encode.
                ["island", "sex"],
            ),
        ],
    )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )


def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
