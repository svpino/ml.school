from metaflow import Runner


def test_register_doesnt_register_model_if_accuracy_under_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=False,
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        accuracy_threshold=0.9,
    ) as running:
        run = running.run
        data = run["register_model"].task.data
        assert data.registered is False, "Model shouldn't have been registered"


def test_register_registers_model_if_accuracy_above_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=False,
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        accuracy_threshold=0.0001,
    ) as running:
        run = running.run
        data = run["register_model"].task.data
        assert data.registered is True, "Model should have been registered"


def test_register_pip_requirements(training_run):
    data = training_run["register_model"].task.data

    assert isinstance(data.pip_requirements, list)
    assert len(data.pip_requirements) > 0
    assert "pandas" in data.pip_requirements


def test_register_signature_inputs(training_run):
    data = training_run["register_model"].task.data

    inputs = [i["name"] for i in data.signature.inputs.to_dict()]

    assert "island" in inputs
    assert "culmen_length_mm" in inputs
    assert "culmen_depth_mm" in inputs
    assert "flipper_length_mm" in inputs
    assert "body_mass_g" in inputs
    assert "sex" in inputs


def test_register_signature_outputs(training_run):
    data = training_run["register_model"].task.data

    outputs = [o["name"] for o in data.signature.outputs.to_dict()]
    assert "prediction" in outputs
    assert "confidence" in outputs


def test_register_signature_params(training_run):
    data = training_run["register_model"].task.data

    params = [p["name"] for p in data.signature.params.to_dict()]
    assert "data_capture" in params


def test_register_artifacts(training_run):
    data = training_run["register_model"].task.data

    assert "model" in data.artifacts
    assert "features_transformer" in data.artifacts
    assert "target_transformer" in data.artifacts