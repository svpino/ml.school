from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3, FlowSpec, Parameter, pypi, pypi_base, step


def load_data_from_s3(location: str):
    print(f"Loading dataset from location {location}")

    with S3(s3root=location) as s3:
        files = s3.get_all()

        print(f"Found {len(files)} file(s) in remote location")

        raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
        return pd.concat(raw_data)


def load_data_from_file():
    location = Path("../penguins.csv")
    print(f"Loading dataset from location {location.as_posix()}")
    return pd.read_csv(location)


def build_model(nodes, learning_rate):
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
    nodes = hp.Int("nodes", 10, 20, step=2)

    learning_rate = hp.Float(
        "learning_rate",
        1e-3,
        1e-2,
        sampling="log",
        default=1e-2,
    )

    return build_model(nodes, learning_rate)


@pypi_base(
    python="3.10.14",
    packages={
        "python-dotenv": "1.0.1",
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.1",
        "numpy": "1.26.4",
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
        self.my_var = "hello world"

        self.next(self.load_data)

    @pypi(packages={"boto3": "1.34.70"})
    @step
    def load_data(self):
        """Load the dataset in memory.

        This function reads every CSV file available and
        concatenates them into a single dataframe.
        """
        import os

        if self.debug:
            df = load_data_from_file()
        else:
            location = f"s3://{os.environ['BUCKET']}/{self.dataset_location}"

            df = load_data_from_s3(location)

        # Shuffle the data
        self.data = df.sample(frac=1, random_state=42)

        print(f"Loaded dataset with {len(self.data)} samples")

        self.next(self.setup_target_transformer)

    @step
    def setup_target_transformer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OrdinalEncoder

        self.target_transformer = ColumnTransformer(
            transformers=[("species", OrdinalEncoder(), [0])],
        )

        self.next(self.setup_features_transformer)

    @step
    def setup_features_transformer(self):
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

        self.features_transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_transformer,
                    make_column_selector(dtype_exclude="object"),
                ),
                ("categorical", categorical_transformer, ["island"]),
            ],
        )

        self.next(self.split_dataset)

    @step
    def split_dataset(self):
        """Split the data into train, validation, and test."""
        from sklearn.model_selection import KFold

        self.target = self.data["species"]
        self.features = self.data.drop("species", axis=1)

        kfold = KFold(n_splits=5, shuffle=True)
        self.folds = list(enumerate(kfold.split(self.target, self.features)))

        self.next(self.transform_target, foreach="folds")

    @step
    def transform_target(self):
        import numpy as np

        self.fold, (self.train_indices, self.test_indices) = self.input

        self.y_train = np.squeeze(
            self.target_transformer.fit_transform(
                np.array(self.target.iloc[self.train_indices]).reshape(-1, 1),
            ),
        )
        self.y_test = np.squeeze(
            self.target_transformer.transform(
                np.array(self.target.iloc[self.test_indices]).reshape(-1, 1),
            ),
        )

        self.next(self.transform_features)

    @step
    def transform_features(self):
        self.x_train = self.features_transformer.fit_transform(
            self.features.iloc[self.train_indices],
        )
        self.x_test = self.features_transformer.transform(
            self.features.iloc[self.test_indices],
        )

        self.next(self.train_model)

    @pypi(
        packages={
            "keras": "3.3.0",
            "jax[cpu]": "0.4.26",
            "packaging": "24.0",
        },
    )
    @step
    def train_model(self):
        print(f"Training fold {self.fold}...")

        self.model = build_model(10, 0.01)

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=50,
            batch_size=32,
            verbose=2,
        )

        self.next(self.evaluate_model)

    @pypi(
        packages={
            "keras": "3.3.0",
            "jax[cpu]": "0.4.26",
            "packaging": "24.0",
        },
    )
    @step
    def evaluate_model(self):
        print(f"Evaluating fold {self.fold}...")

        self.loss, self.accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2,
        )

        print(f"Fold {self.fold} - loss: {self.loss} - accuracy: {self.accuracy}")
        self.next(self.reduce)

    @step
    def reduce(self, inputs):
        import numpy as np

        accuracies = [i.accuracy for i in inputs]
        accuracy = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        print(f"Accuracy: {accuracy} +-{accuracy_std}")

        self.next(self.end)

    @step
    def end(self):
        print("the end")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TrainingFlow()
