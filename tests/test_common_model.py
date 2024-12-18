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
