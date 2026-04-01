"""Tests for the parse_project_configuration function in common/pipeline.py."""

import os
from unittest.mock import patch

from common.pipeline import parse_project_configuration


class TestParseProjectConfiguration:
    """Tests for the parse_project_configuration function."""

    def test_parse_config_adds_default_mlflow_uri(self):
        """Missing mlflow_tracking_uri should default to http://localhost:5000."""
        with patch.dict(os.environ, {}, clear=True):
            config = parse_project_configuration("backend:\n  module: backend.Local")

        assert config["mlflow_tracking_uri"] == "http://localhost:5000"

    def test_parse_config_preserves_existing_mlflow_uri(self):
        """An explicit mlflow_tracking_uri in the YAML should be kept as-is."""
        yaml_content = (
            "mlflow_tracking_uri: http://custom:8080\nbackend:\n  module: backend.Local"
        )
        config = parse_project_configuration(yaml_content)

        assert config["mlflow_tracking_uri"] == "http://custom:8080"

    def test_parse_config_adds_default_backend(self):
        """Missing backend should default to {'module': 'backend.Local'}."""
        config = parse_project_configuration(
            "mlflow_tracking_uri: http://localhost:5000"
        )

        assert config["backend"] == {"module": "backend.Local"}

    def test_parse_config_expands_env_vars_in_backend(self):
        """Environment variables in backend values should be expanded."""
        yaml_content = "backend:\n  path: ${MY_VAR}/data"

        with patch.dict(os.environ, {"MY_VAR": "/opt"}):
            config = parse_project_configuration(yaml_content)

        assert config["backend"]["path"] == "/opt/data"

    def test_parse_config_leaves_unexpanded_env_var_when_not_set(self):
        """Unset environment variables should remain as literal ${VAR} placeholders."""
        yaml_content = "backend:\n  path: ${MISSING_VAR}/data"

        with patch.dict(os.environ, {}, clear=True):
            config = parse_project_configuration(yaml_content)

        assert config["backend"]["path"] == "${MISSING_VAR}/data"

    def test_parse_config_uses_mlflow_tracking_uri_env_var(self):
        """Missing mlflow_tracking_uri should fall back to the MLFLOW_TRACKING_URI env var."""
        with patch.dict(
            os.environ, {"MLFLOW_TRACKING_URI": "http://remote:9000"}, clear=True
        ):
            config = parse_project_configuration("backend:\n  module: backend.Local")

        assert config["mlflow_tracking_uri"] == "http://remote:9000"

    def test_parse_config_skips_env_expansion_for_non_string_backend_values(self):
        """Non-string backend values should pass through without error."""
        yaml_content = (
            "backend:\n  module: backend.Local\n  retries: 3\n  enabled: true"
        )
        config = parse_project_configuration(yaml_content)

        assert config["backend"]["retries"] == 3
        assert config["backend"]["enabled"] is True
