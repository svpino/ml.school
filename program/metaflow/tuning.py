import os

from common import (
    PACKAGES,
    build_features_transformer,
    build_model,
    build_target_transformer,
    load_dataset,
)
from metaflow import (
    FlowSpec,
    IncludeFile,
    card,
    current,
    project,
    pypi,
    pypi_base,
    retry,
    step,
)


def build_tuner_model(hp):
    learning_rate = hp.Float(
        "learning_rate",
        1e-3,
        1e-2,
        sampling="log",
        default=1e-2,
    )

    return build_model(learning_rate)


@project(name="penguins")
@pypi_base(
    python="3.10.14",
    packages=PACKAGES,
)
class TuningFlow(FlowSpec):
    dataset = IncludeFile(
        "penguins",
        is_text=True,
        help="Penguins dataset",
        default="../penguins.csv",
    )

    @step
    def start(self):
        self.next(self.load_data)

    @pypi(packages={"boto3": "1.34.70"})
    @retry
    @card
    @step
    def load_data(self):
        """Load the dataset in memory."""
        dataset = os.environ["DATASET"] if current.is_production else self.dataset
        self.data = load_dataset(
            dataset,
            is_production=current.is_production,
        )

        print(f"Loaded dataset with {len(self.data)} samples")

        self.next(self.split_dataset)

    @step
    def split_dataset(self):
        """Split the data into train, validation, and test."""
        from sklearn.model_selection import train_test_split

        self.df_train, temp = train_test_split(self.data, test_size=0.3)
        self.df_validation, self.df_test = train_test_split(temp, test_size=0.5)

        self.next(self.transform)

    @step
    def transform(self):
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            self.df_train.species.to_numpy().reshape(-1, 1),
        )
        self.y_validation = target_transformer.transform(
            self.df_validation.species.to_numpy().reshape(-1, 1),
        )
        self.y_test = target_transformer.transform(
            self.df_test.species.to_numpy().reshape(-1, 1),
        )

        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(self.df_train)
        self.x_validation = features_transformer.transform(self.df_validation)
        self.x_test = features_transformer.transform(self.df_test)

        print(f"Train samples: {len(self.x_train)}")
        print(f"Validation samples: {len(self.x_validation)}")
        print(f"Test samples: {len(self.x_test)}")

        self.next(self.tune_model)

    @pypi(
        packages={
            "packaging": "24.0",
            "keras-tuner": "1.4.7",
            "grpcio": "1.62.1",
            "protobuf": "4.25.3",
        },
    )
    @step
    def tune_model(self):
        from keras_tuner import RandomSearch

        tuner = RandomSearch(
            hypermodel=build_tuner_model,
            objective="val_accuracy",
            max_trials=5,
            executions_per_trial=2,
            overwrite=True,
            directory=".metaflow",
            project_name="tuning",
        )

        tuner.search_space_summary()

        tuner.search(
            self.x_train,
            self.y_train,
            validation_data=(self.x_validation, self.y_validation),
            batch_size=32,
            epochs=50,
            verbose=2,
        )

        tuner.results_summary()

        hyperparameters = tuner.get_best_hyperparameters()[0]

        self.learning_rate = hyperparameters.get("learning_rate")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TuningFlow()
