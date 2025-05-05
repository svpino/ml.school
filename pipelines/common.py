import logging
import logging.config
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import IncludeFile, current

PYTHON = "3.12.8"

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.2",
    "tensorflow": "2.18.0",
    "evidently": "0.6.7"
}


class Pipeline:
    """A base class for all pipelines."""

    def configure_logging(self) -> logging.Logger:
        """Configure the logging handler and return a logger instance."""
        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )

        return logging.getLogger("mlschool")


class DatasetMixin:
    """A mixin for loading and preparing a dataset.

    This mixin is designed to be combined with any pipeline that requires accessing
    a dataset.
    """

    dataset = IncludeFile(
        "dataset",
        is_text=True,
        help="Dataset that will be used to train the model.",
        default="data/penguins.csv",
    )

    def load_dataset(self, logger=None):
        """Load and prepare the dataset.

        This method loads the dataset, cleans the sex column by replacing extraneous values with NaN,
        drops any rows with missing values, and then shuffles the dataset.
        """
        import numpy as np

        # The raw data is passed as a string, so we need to convert it into a DataFrame.
        data = pd.read_csv(StringIO(self.dataset))

        # Replace extraneous values in the sex column with NaN
        data["sex"] = data["sex"].replace(".", np.nan)

        # Drop rows with missing values
        row_count_before = len(data)
        data = data.dropna()
        if logger:
            logger.info("Dropped %d rows with missing values",
                        row_count_before - len(data))
        else:
            logging.info("Dropped %d rows with missing values",
                         row_count_before - len(data))

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        if logger:
            logger.info("Loaded dataset with %d samples", len(data))

        return data


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
