import logging
import time
from io import StringIO

from metaflow import S3

PYTHON = "3.12"

PACKAGES = {
    "python-dotenv": "1.0.1",
    "scikit-learn": "1.5.1",
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "keras": "3.5.0",
    "jax[cpu]": "0.4.31",
    "packaging": "24.1",
    "mlflow": "2.16.0",
    "setuptools": "74.1.2",
}

TRAINING_EPOCHS = 50
TRAINING_BATCH_SIZE = 32

logger = logging.getLogger(__name__)


def load_dataset(dataset: str, *, is_production: bool = False):
    """Load and prepare the dataset.

    When running in production mode, this function reads every CSV file available in the
    supplied S3 location and concatenates them into a single dataframe. When running in
    development mode, this function reads the dataset from the supplied string
    parameter.
    """
    import numpy as np
    import pandas as pd

    if is_production:
        # Load the dataset from an S3 location.
        with S3(s3root=dataset) as s3:
            files = s3.get_all()

            logger.info("Found %d file(s) in remote location", len(files))

            raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
            data = pd.concat(raw_data)
    else:
        # When running in development mode, the raw data is passed as a string, so we
        # can convert it to a DataFrame.
        data = pd.read_csv(StringIO(dataset))

    # Replace extraneous data in the sex column with NaN. We can handle missing values
    # later in the pipeline.
    data["sex"] = data["sex"].replace(".", np.nan)

    # We want to shuffle the dataset. For reproducibility, we can fix the seed value
    # when running in development mode. When running in production mode, we can use
    # the current time as the seed to ensure a different shuffle each time the pipeline
    # is executed.
    seed = int(time.time() * 1000) if is_production else 42
    generator = np.random.default_rng(seed=seed)
    return data.sample(frac=1, random_state=generator)


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
