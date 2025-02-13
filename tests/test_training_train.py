from keras import optimizers

from pipelines.common import build_model


def test_build_model_configures_input_layer_correctly():
    model = build_model(input_shape=9)
    assert model.input_shape == (None, 9)

    model = build_model(input_shape=3)
    assert model.input_shape == (None, 3)


def test_build_model_configures_output_layer_correctly():
    model = build_model(input_shape=9)
    assert model.output_shape == (None, 3)

    output_layer = model.layers[-1]
    assert output_layer.activation.__name__ == "softmax"


def test_build_model_configures_optimizer_correctly():
    learning_rate = 0.1

    model = build_model(input_shape=9, learning_rate=learning_rate)
    assert isinstance(model.optimizer, optimizers.SGD)
    assert model.optimizer.learning_rate == learning_rate

    learning_rate = 0.01
    model = build_model(input_shape=9, learning_rate=learning_rate)
    assert model.optimizer.learning_rate == learning_rate


def test_build_model_loss_function():
    model = build_model(input_shape=9)
    assert model.loss == "sparse_categorical_crossentropy"


def test_train_fold_builds_model(training_run):
    data = training_run["train_fold"].task.data
    assert data.model is not None


def test_train_fold_creates_mlflow_nested_run(training_run):
    data = training_run["train_fold"].task.data
    assert data.mlflow_fold_run_id is not None


def test_train_stores_model_as_artifact(training_run):
    data = training_run["train"].task.data
    assert data.model is not None
