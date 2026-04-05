import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from inference.backend import Local


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def backend(db_path):
    return Local(config={"database": db_path})


@pytest.fixture
def backend_with_logger(db_path):
    logger = logging.getLogger("test")
    return Local(config={"database": db_path}, logger=logger)


def _sample_input():
    return pd.DataFrame(
        [
            {
                "island": "Torgersen",
                "culmen_length_mm": 39.1,
                "culmen_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": "MALE",
            },
        ],
    )


def _sample_output():
    return [{"prediction": "Adelie", "confidence": 0.95}]


# --- Local.__init__ ---


def test_init_uses_config_database(db_path):
    backend = Local(config={"database": db_path})
    assert backend.database == db_path


def test_init_uses_config_target():
    backend = Local(config={"target": "http://custom:9090/predict"})
    assert backend.target == "http://custom:9090/predict"


def test_init_defaults_target_without_config():
    backend = Local()
    assert backend.target == "http://127.0.0.1:8080/invocations"


def test_init_defaults_database_without_config():
    backend = Local()
    assert backend.database == "data/penguins.db"


def test_init_reads_database_from_env(monkeypatch, tmp_path):
    db = str(tmp_path / "env.db")
    monkeypatch.setenv("MODEL_BACKEND_DATABASE", db)
    backend = Local()
    assert backend.database == db


# --- Local.load ---


def test_load_returns_none_when_database_missing(tmp_path):
    backend = Local(config={"database": str(tmp_path / "nonexistent.db")})
    assert backend.load() is None


def test_load_returns_saved_data(backend):
    backend.save(_sample_input(), _sample_output())
    data = backend.load()

    assert len(data) == 1
    assert data.iloc[0]["island"] == "Torgersen"
    assert data.iloc[0]["prediction"] == "Adelie"


def test_load_respects_limit(backend):
    for _ in range(5):
        backend.save(_sample_input(), _sample_output())

    data = backend.load(limit=3)
    assert len(data) == 3


# --- Local.save ---


def test_save_creates_database(backend, db_path):
    backend.save(_sample_input(), _sample_output())

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    conn.close()

    assert rows == 1


def test_save_stores_prediction_and_confidence(backend, db_path):
    backend.save(_sample_input(), _sample_output())

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prediction, confidence FROM data").fetchone()
    conn.close()

    assert row[0] == "Adelie"
    assert row[1] == 0.95


def test_save_sets_null_prediction_for_empty_output(backend, db_path):
    backend.save(_sample_input(), [])

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prediction, confidence, target FROM data").fetchone()
    conn.close()

    assert row[0] is None
    assert row[1] is None
    assert row[2] is None


def test_save_sets_null_prediction_for_none_output(backend, db_path):
    backend.save(_sample_input(), None)

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prediction, confidence FROM data").fetchone()
    conn.close()

    assert row[0] is None
    assert row[1] is None


def test_save_generates_uuid(backend, db_path):
    backend.save(_sample_input(), _sample_output())

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT uuid FROM data").fetchone()
    conn.close()

    assert row[0] is not None
    assert len(row[0]) == 36  # UUID format


def test_save_appends_to_existing_data(backend, db_path):
    backend.save(_sample_input(), _sample_output())
    backend.save(_sample_input(), _sample_output())

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    conn.close()

    assert rows == 2


# --- Local.label ---


def test_label_returns_zero_when_database_missing(tmp_path):
    backend = Local(config={"database": str(tmp_path / "nonexistent.db")})
    assert backend.label() == 0


def test_label_returns_zero_when_no_unlabeled_samples(backend):
    backend.save(_sample_input(), _sample_output())

    # Manually set a target so there are no unlabeled samples
    conn = sqlite3.connect(backend.database)
    conn.execute("UPDATE data SET target = 'Adelie'")
    conn.commit()
    conn.close()

    assert backend.label() == 0


def test_label_labels_unlabeled_samples(backend):
    backend.save(_sample_input(), _sample_output())

    count = backend.label(ground_truth_quality=1.0)
    assert count == 1

    data = backend.load()
    assert data.iloc[0]["target"] is not None


def test_label_with_perfect_quality_matches_prediction(backend):
    backend.save(_sample_input(), _sample_output())

    backend.label(ground_truth_quality=1.0)

    conn = sqlite3.connect(backend.database)
    row = conn.execute("SELECT prediction, target FROM data").fetchone()
    conn.close()

    assert row[0] == row[1]  # prediction == target when quality=1.0


# --- Backend.get_fake_label ---


def test_get_fake_label_returns_prediction_with_perfect_quality(backend):
    result = backend.get_fake_label("Adelie", ground_truth_quality=1.0)
    assert result == "Adelie"


def test_get_fake_label_returns_random_with_zero_quality(backend):
    results = {
        backend.get_fake_label("Adelie", ground_truth_quality=0.0)
        for _ in range(100)
    }
    # With quality=0.0, we should get random species from the set
    assert results <= {"Adelie", "Chinstrap", "Gentoo"}


# --- Backend._log / _info / _error / _exception ---


def test_log_info_calls_logger(backend_with_logger):
    backend_with_logger.logger = MagicMock()
    backend_with_logger._info("test message")
    backend_with_logger.logger.info.assert_called_once_with("test message")


def test_log_error_calls_logger(backend_with_logger):
    backend_with_logger.logger = MagicMock()
    backend_with_logger._error("error message")
    backend_with_logger.logger.error.assert_called_once_with("error message")


def test_log_exception_calls_logger(backend_with_logger):
    backend_with_logger.logger = MagicMock()
    backend_with_logger._exception("exception message")
    backend_with_logger.logger.exception.assert_called_once_with("exception message")


def test_log_does_nothing_without_logger(backend):
    backend.logger = None
    # Should not raise
    backend._info("no logger")
    backend._error("no logger")
    backend._exception("no logger")


# --- Local.invoke ---


def test_invoke_posts_to_target(backend):
    mock_response = MagicMock()
    mock_response.json.return_value = {"predictions": ["Adelie"]}

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = backend.invoke([{"island": "Torgersen"}])

    assert result == {"predictions": ["Adelie"]}
    mock_post.assert_called_once()


def test_invoke_returns_none_on_error(backend):
    with patch("requests.post", side_effect=ConnectionError("refused")):
        result = backend.invoke([{"island": "Torgersen"}])

    assert result is None
