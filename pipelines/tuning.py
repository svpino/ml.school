from common import (
    PACKAGES,
    PYTHON,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    FlowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
)
from metaflow import (
    FlowSpec,
    project,
    pypi,
    pypi_base,
    step,
)


def build_tuner_model(hp):
    """Build a hyperparameter-tunable model."""
    # TODO: Document this
    learning_rate = hp.Float(
        "learning_rate",
        1e-3,
        1e-2,
        sampling="log",
        default=1e-2,
    )

    # The input of the model is a vector of 9 values: four values to
    # represent the numerical features (culmen_length_mm, culmen_depth_mm,
    # flipper_length_mm, and body_mass) and five values to represent the
    # categorical features (island and sex) encoded as a one-hot vector.
    return build_model(input_shape=9, learning_rate=learning_rate)


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=PACKAGES,
)
# TODO: Quitar packages de aqui y usar la funcion packages()
class TuningFlow(FlowSpec, FlowMixin):
    @step
    def start(self):
        self.training_parameters = {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }

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
            verbose=2,
            **self.training_parameters,
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
