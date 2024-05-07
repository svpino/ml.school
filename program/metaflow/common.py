from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3


def load_data_from_s3(location: str):
    """Load the dataset from an S3 location.

    This function will concatenate every CSV file in the given location
    and return a single DataFrame.
    """
    print(f"Loading dataset from location {location}")

    with S3(s3root=location) as s3:
        files = s3.get_all()

        print(f"Found {len(files)} file(s) in remote location")

        raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
        return pd.concat(raw_data)


def load_data_from_file():
    """Load the dataset from a local file.

    This function is useful to test the pipeline locally
    without having to access the data remotely.
    """
    location = Path("../penguins.csv")
    print(f"Loading dataset from location {location.as_posix()}")
    return pd.read_csv(location)


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
        OneHotEncoder(),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_exclude="object"),
            ),
            ("categorical", categorical_transformer, ["island"]),
        ],
    )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target variable."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])],
    )


def build_model(nodes, learning_rate):
    """Build and compile a simple neural network."""
    from keras import Input
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD

    model = Sequential(
        [
            Input(shape=(7,)),
            Dense(nodes, activation="relu"),
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


def build_tuner_model(hp):
    """Build a hyperparameter-tunable model."""
    nodes = hp.Int("nodes", 10, 20, step=2)

    learning_rate = hp.Float(
        "learning_rate",
        1e-3,
        1e-2,
        sampling="log",
        default=1e-2,
    )

    return build_model(nodes, learning_rate)
