import json
import os
from unittest.mock import Mock, patch

import pytest

from inference.model import Model


class TestModelBackend:
    """Tests for the Model backend initialization."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create a temporary config file."""
        config_file = tmp_path / "config.json"
        config_data = {"database": "penguins-test.db"}
        config_file.write_text(json.dumps(config_data))
        return config_file

    def test_model_initializes_with_no_backend(self):
        """The backend should be None before load_context is called."""
        model = Model()
        assert model.backend is None

    def test_backend_is_loaded_from_environment_variable(self):
        """load_context should import and instantiate the class from MODEL_BACKEND."""
        with (
            patch.dict(os.environ, {"MODEL_BACKEND": "module.Backend"}),
            patch("importlib.import_module") as mock_import,
        ):
            mock_backend = Mock()
            mock_backend_class = Mock(return_value=mock_backend)

            mock_module = Mock()
            mock_module.Backend = mock_backend_class
            mock_import.return_value = mock_module

            model = Model()
            model.load_context(context=None)

            assert model.backend is mock_backend
            mock_import.assert_called_once_with("module")

    def test_backend_is_initialized_with_config_file(self, temp_config):
        """A valid MODEL_BACKEND_CONFIG file should be parsed and passed to the backend."""
        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_BACKEND": "module.Backend",
                    "MODEL_BACKEND_CONFIG": str(temp_config),
                },
            ),
            patch("importlib.import_module") as mock_import,
        ):
            mock_backend = Mock()
            mock_backend_class = Mock(return_value=mock_backend)

            mock_module = Mock()
            mock_module.Backend = mock_backend_class
            mock_import.return_value = mock_module

            model = Model()
            model.load_context(context=None)

            assert model.backend is mock_backend
            mock_backend_class.assert_called_once_with(
                config={"database": "penguins-test.db"},
            )

    def test_backend_is_initialized_with_missing_config_file(self):
        """A missing MODEL_BACKEND_CONFIG file should result in config=None."""
        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_BACKEND": "module.Backend",
                    "MODEL_BACKEND_CONFIG": "nonexistent.json",
                },
            ),
            patch("importlib.import_module") as mock_import,
        ):
            mock_backend = Mock()
            mock_backend_class = Mock(return_value=mock_backend)

            mock_module = Mock()
            mock_module.Backend = mock_backend_class
            mock_import.return_value = mock_module

            model = Model()
            model.load_context(context=None)

            assert model.backend is mock_backend
            mock_backend_class.assert_called_once_with(config=None)

    def test_backend_is_initialized_with_invalid_config(self, tmp_path):
        """Malformed JSON in MODEL_BACKEND_CONFIG should set the backend to None."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("{not valid json")

        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_BACKEND": "module.Backend",
                    "MODEL_BACKEND_CONFIG": str(config_file),
                },
            ),
            patch("importlib.import_module"),
        ):
            model = Model()
            model.load_context(context=None)

            assert model.backend is None

    def test_backend_is_set_to_none_if_class_doesnt_exist(self):
        """An unimportable MODEL_BACKEND should set the backend to None."""
        with patch.dict(
            os.environ, {"MODEL_BACKEND": "nonexistent_module.NonexistentClass"}
        ):
            model = Model()
            model.load_context(context=None)
            assert model.backend is None


@pytest.fixture
def backend():
    """Return a Local backend instance."""
    from inference.backend import Local

    return Local(config={"database": ":memory:"}, logger=None)


class TestGetFakeLabel:
    """Tests for the get_fake_label method."""

    def test_get_fake_label_returns_prediction_when_quality_is_1(self, backend):
        """Quality of 1.0 should always return the original prediction."""
        with patch("random.random", return_value=0.5):
            result = backend.get_fake_label("Adelie", 1.0)
            assert result == "Adelie"

    def test_get_fake_label_returns_random_species_when_quality_is_0(self, backend):
        """Quality of 0.0 should always return a random species."""
        with (
            patch("random.random", return_value=0.5),
            patch("random.choice", return_value="Gentoo"),
        ):
            result = backend.get_fake_label("Adelie", 0.0)
            assert result == "Gentoo"

    def test_get_fake_label_returns_prediction_when_random_below_quality(self, backend):
        """A random value below the quality threshold should return the prediction."""
        with patch("random.random", return_value=0.3):
            result = backend.get_fake_label("Adelie", 0.5)
            assert result == "Adelie"

    def test_get_fake_label_returns_random_species_when_random_above_quality(
        self, backend
    ):
        """A random value above the quality threshold should pick a random species."""
        with (
            patch("random.random", return_value=0.7),
            patch("random.choice", return_value="Chinstrap") as mock_choice,
        ):
            result = backend.get_fake_label("Adelie", 0.5)
            assert result == "Chinstrap"
            mock_choice.assert_called_once_with(
                ["Adelie", "Chinstrap", "Gentoo"],
            )


class TestLog:
    """Tests for the _log method."""

    def test_log_dispatches_to_info_level(self, backend):
        """Level 'info' should delegate to logger.info."""
        mock_logger = Mock()
        backend.logger = mock_logger

        backend._log("test message", "info")

        mock_logger.info.assert_called_once_with("test message")

    def test_log_dispatches_to_error_level(self, backend):
        """Level 'error' should delegate to logger.error."""
        mock_logger = Mock()
        backend.logger = mock_logger

        backend._log("test message", "error")

        mock_logger.error.assert_called_once_with("test message")

    def test_log_dispatches_to_exception_level(self, backend):
        """Level 'exception' should delegate to logger.exception."""
        mock_logger = Mock()
        backend.logger = mock_logger

        backend._log("test message", "exception")

        mock_logger.exception.assert_called_once_with("test message")

    def test_log_ignores_unsupported_level(self, backend):
        """An unsupported level should silently ignore the message."""
        mock_logger = Mock()
        backend.logger = mock_logger

        backend._log("test message", "debug")

        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.exception.assert_not_called()

    def test_log_does_nothing_without_logger(self, backend):
        """A None logger should not raise any exception."""
        backend.logger = None
        backend._log("test message", "info")
