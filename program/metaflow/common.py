import logging
import os
import time
from io import StringIO

from metaflow import S3, IncludeFile, current

PYTHON = "3.12"

PACKAGES = {
    "python-dotenv": "1.0.1",
    "scikit-learn": "1.5.1",
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "keras": "3.5.0",
    "jax[cpu]": "0.4.31",
    "tensorflow": "2.17.0",
    "boto3": "1.35.15",
    "packaging": "24.1",
    "mlflow": "2.16.0",
    "setuptools": "74.1.2",
}

TRAINING_EPOCHS = 50
TRAINING_BATCH_SIZE = 32

logger = logging.getLogger(__name__)


class FlowMixin:
    dataset = IncludeFile(
        "penguins",
        is_text=True,
        help=(
            "Local copy of the penguins dataset. This file will be included in the "
            "flow and will be used whenever the flow is executed in development mode."
        ),
        default="../penguins.csv",
    )

    # @property
    # def data(self):
    #     print("Getting data", self._data)
    #     if not self._data:
    #         print("Loading dataset")
    #         dataset = (
    #             os.environ.get("DATASET", self.dataset)
    #             if current.is_production
    #             else self.dataset
    #         )

    #         # Load the dataset in memory. This function will either read the dataset from
    #         # the included file or from an S3 location, depending on the mode in which the
    #         # flow is running.
    #         self._data = self.load_dataset(dataset)

    #         logging.info("Loaded dataset with %d samples", len(self._data))

    #     return self._data

    def load_dataset(self):
        """Load and prepare the dataset.

        When running in production mode, this function reads every CSV file available in
        the supplied S3 location and concatenates them into a single dataframe. When
        running in development mode, this function reads the dataset from the supplied
        string parameter.
        """
        import numpy as np
        import pandas as pd

        if current.is_production:
            dataset = os.environ.get("DATASET", self.dataset)

            with S3(s3root=dataset) as s3:
                files = s3.get_all()

                logger.info("Found %d file(s) in remote location", len(files))

                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
                data = pd.concat(raw_data)
        else:
            # When running in development mode, the raw data is passed as a string,
            # so we can convert it to a DataFrame.
            data = pd.read_csv(StringIO(self.dataset))

        # Replace extraneous data in the sex column with NaN. We can handle missing
        # values later in the pipeline.
        data["sex"] = data["sex"].replace(".", np.nan)

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])],
    )


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


def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD

    model = Sequential(
        [
            Input(shape=(input_shape,)),
            Dense(10, activation="relu"),
            Dense(8, activation="relu"),
            Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
