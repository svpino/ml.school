from common.pipeline import parse_project_configuration


def test_parse_returns_config_from_yaml():
    yaml_input = (
        "project: penguins\n"
        "mlflow_tracking_uri: http://remote:5000\n"
        "backend:\n"
        "  module: backend.Sagemaker\n"
    )
    config = parse_project_configuration(yaml_input)

    assert config["project"] == "penguins"
    assert config["mlflow_tracking_uri"] == "http://remote:5000"
    assert config["backend"]["module"] == "backend.Sagemaker"


def test_parse_defaults_mlflow_tracking_uri_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-uri:9999")

    yaml_input = "project: penguins\nbackend:\n  module: backend.Local\n"
    config = parse_project_configuration(yaml_input)

    assert config["mlflow_tracking_uri"] == "http://env-uri:9999"


def test_parse_defaults_mlflow_tracking_uri_fallback(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    yaml_input = "project: penguins\nbackend:\n  module: backend.Local\n"
    config = parse_project_configuration(yaml_input)

    assert config["mlflow_tracking_uri"] == "http://localhost:5000"


def test_parse_defaults_backend_when_missing():
    yaml_input = "project: penguins\n"
    config = parse_project_configuration(yaml_input)

    assert config["backend"] == {"module": "backend.Local"}


def test_parse_expands_env_vars_in_backend_values(monkeypatch):
    monkeypatch.setenv("MY_BUCKET", "s3://prod-bucket")

    yaml_input = (
        "project: penguins\n"
        "backend:\n"
        "  module: backend.Sagemaker\n"
        "  data-capture-uri: ${MY_BUCKET}/datastore\n"
    )
    config = parse_project_configuration(yaml_input)

    assert config["backend"]["data-capture-uri"] == "s3://prod-bucket/datastore"


def test_parse_preserves_unexpanded_env_vars(monkeypatch):
    monkeypatch.delenv("UNDEFINED_VAR", raising=False)

    yaml_input = (
        "project: penguins\n"
        "backend:\n"
        "  module: backend.Local\n"
        "  target: ${UNDEFINED_VAR}\n"
    )
    config = parse_project_configuration(yaml_input)

    assert config["backend"]["target"] == "${UNDEFINED_VAR}"


def test_parse_skips_non_string_backend_values():
    yaml_input = (
        "project: penguins\n"
        "backend:\n"
        "  module: backend.Local\n"
        "  retries: 3\n"
    )
    config = parse_project_configuration(yaml_input)

    assert config["backend"]["retries"] == 3
