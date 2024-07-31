from io import StringIO

from metaflow import S3

PACKAGES = {
    "python-dotenv": "1.0.1",
    "scikit-learn": "1.5.0",
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "keras": "3.3.3",
    "jax[cpu]": "0.4.28",
    "packaging": "24.1",
    "mlflow": "2.15.0",
    "setuptools": "72.1.0",
}


def load_dataset(dataset: str, *, is_production: bool = False):
    """Load and prepare the dataset.

    When running in production mode, this function reads every CSV file
    available in the supplied S3 location and concatenates them into a
    single dataframe. When running in development mode, it reads the
    dataset from the supplied string parameter.
    """
    import numpy as np
    import pandas as pd

    if is_production:
        # Load the dataset from an S3 location.
        with S3(s3root=dataset) as s3:
            files = s3.get_all()

            print(f"Found {len(files)} file(s) in remote location")

            raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
            data = pd.concat(raw_data)
    else:
        # When running in development mode, the raw data is passed
        # as a string, so we can convert it to a DataFrame.
        data = pd.read_csv(StringIO(dataset))

    # Replace extraneous data in the sex column with NaN.
    data["sex"] = data["sex"].replace(".", np.nan)

    # Shuffle the dataset.
    # TODO: Use seed only when development mode
    data = data.sample(frac=1, random_state=42)

    return data


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target variable."""
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
        # We can use the `handle_unknown="ignore"` parameter to ignore
        # unseen categories during inference. When encoding an unknown
        # category, the transformer will return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which
                # is also a categorical column. To accomplish this, we
                # can specify the column names we want to encode.
                ["island", "sex"],
            ),
        ],
    )


def build_model(learning_rate=0.01):
    """Build and compile a simple neural network to predict the species of a penguin."""
    from keras import Input
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD

    model = Sequential(
        [
            Input(shape=(9,)),
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
