import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

from pipelines.common import build_features_transformer, build_target_transformer


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame(
        {
            "species": ["Adelie", "Gentoo", "Chinstrap"],
            "island": ["Torgersen", "Biscoe", "Dream"],
            "culmen_length_mm": [39.1, 39.5, 40.1],
            "culmen_depth_mm": [18.7, 17.4, 18.0],
            "flipper_length_mm": [181, 186, 195],
            "body_mass_g": [3750, 3800, 3650],
            "sex": ["MALE", "FEMALE", "MALE"],
        },
    )


def test_build_features_transformer_numeric_columns():
    transformer = build_features_transformer()

    assert "numeric" in transformer.transformers[0][0]

    steps = [step[0] for step in transformer.transformers[0][1].steps]
    assert "simpleimputer" in steps
    assert "standardscaler" in steps


def test_build_features_transformer_categorical_columns():
    transformer = build_features_transformer()

    assert "categorical" in transformer.transformers[1][0]

    steps = [step[0] for step in transformer.transformers[1][1].steps]
    assert "simpleimputer" in steps
    assert "onehotencoder" in steps


def test_features_transformer_returns_correct_number_of_columns(data):
    transformer = build_features_transformer()
    transformed_data = transformer.fit_transform(data)

    # After transforming the data, the number of features should be 9:
    # 3 - island (one-hot encoded)
    # 2 - sex (one-hot encoded)
    # 1 - culmen_length_mm
    # 1 - culmen_depth_mm
    # 1 - flipper_length_mm
    # 1 - body_mass_g
    number_of_columns = 9

    assert transformed_data.shape == (
        len(data),
        number_of_columns,
    ), "Unexpected output shape after transforming the dataset"


def test_build_target_transformer_species():
    transformer = build_target_transformer()

    assert "species" in transformer.transformers[0][0]
    assert isinstance(transformer.transformers[0][1], OrdinalEncoder)


def test_target_transformer_returns_correct_number_of_columns(data):
    transformer = build_target_transformer()
    transformed_data = transformer.fit_transform(data)
    assert transformed_data.shape == (
        len(data),
        1,
    ), "Unexpected output shape after transforming the target column"


def test_transform_fold_sets_fold_index(training_run):
    data = training_run["transform_fold"].task.data
    assert data.fold in range(5)


def test_transform_fold_processes_data_splits(training_run):
    data = training_run["transform_fold"].task.data

    train_size = len(data.train_indices)
    test_size = len(data.test_indices)

    assert data.x_train.shape == (train_size, 9)
    assert data.y_train.shape == (train_size, 1)

    assert data.x_test.shape == (test_size, 9)
    assert data.y_test.shape == (test_size, 1)


def test_transform_processes_dataset(training_run):
    data = training_run["transform"].task.data

    dataset_size = len(data.data)

    assert data.x.shape == (dataset_size, 9)
    assert data.y.shape == (dataset_size, 1)


def test_transform_stores_transformation_pipelines_as_artifacts(training_run):
    data = training_run["transform"].task.data

    assert data.features_transformer is not None
    assert data.target_transformer is not None
