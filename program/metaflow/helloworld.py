from metaflow import S3, FlowSpec, Parameter, pypi, pypi_base, step


@pypi_base(
    python="3.10.14",
    packages={
        "python-dotenv": "1.0.1",
        "scikit-learn": "1.4.1.post1",
        "pandas": "2.2.1",
        "numpy": "1.26.4",
    },
)
class PenguinsDataProcessingFlow(FlowSpec):
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
        from io import StringIO

        import pandas as pd

        location = f"s3://{os.environ['BUCKET']}/{self.dataset_location}"
        print(f"Loading dataset from location {location}")

        with S3(s3root=location) as s3:
            files = s3.get_all()

            print(f"Found {len(files)} file(s) in remote location")

            raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
            df = pd.concat(raw_data)

        # Shuffle the data
        self.data = df.sample(frac=1, random_state=42)

        print(f"Loaded dataset with {len(self.data)} samples")

        self.next(self.split_dataset)

    @step
    def split_dataset(self):
        """Split the data into train, validation, and test."""
        from sklearn.model_selection import train_test_split

        self.df_train, temp = train_test_split(self.data, test_size=0.3)
        self.df_validation, self.df_test = train_test_split(temp, test_size=0.5)

        self.next(self.setup_pipeline)

    @step
    def setup_pipeline(self):
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        self.target_transformer = ColumnTransformer(
            transformers=[("species", OrdinalEncoder(), [0])],
        )

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

        self.next(self.transform_data)

    @step
    def transform_data(self):
        import numpy as np

        self.y_train = self.target_transformer.fit_transform(
            np.array(self.df_train.species.values).reshape(-1, 1),
        )
        self.y_validation = self.target_transformer.transform(
            np.array(self.df_validation.species.values).reshape(-1, 1),
        )
        self.y_test = self.target_transformer.transform(
            np.array(self.df_test.species.values).reshape(-1, 1),
        )

        self.df_train = self.df_train.drop("species", axis=1)
        self.df_validation = self.df_validation.drop("species", axis=1)
        self.df_test = self.df_test.drop("species", axis=1)

        self.X_train = self.features_transformer.fit_transform(self.df_train)
        self.X_validation = self.features_transformer.transform(self.df_validation)
        self.X_test = self.features_transformer.transform(self.df_test)

        print(f"Train samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_validation)}")
        print(f"Test samples: {len(self.X_test)}")

        self.next(self.end)

    @step
    def end(self):
        print("the data artifact is still: %s" % self.my_var)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    PenguinsDataProcessingFlow()
