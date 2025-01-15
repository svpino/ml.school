import importlib
import logging
import os
import random
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from metaflow import Config, Parameter


class BackendMixin:
    """A mixin for managing the backend implementation of a model.

    This mixin is designed to be combined with any pipeline that requires accessing
    the backend database of a hosted model.
    """

    backend_config = Config(
        "backend",
        help=("Backend configuration used to initialize the provided backend class."),
        default={},
    )

    backend = Parameter(
        "backend",
        help=(
            "Class implementing the `backend.Backend` abstract class. This "
            "class is responsible for storing and loading data from the database "
            "backing the hosted model."
        ),
        default="backend.SQLite",
    )

    def load_backend(self):
        """Instantiate the backend class using the supplied configuration."""
        try:
            module, cls = self.backend.rsplit(".", 1)
            module = importlib.import_module(module)
            backend_impl = getattr(module, cls)(config=self.backend_config)
        except Exception as e:
            message = f"There was an error instantiating class {self.backend}."
            raise RuntimeError(message) from e
        else:
            logging.info("Backend: %s", self.backend)
            return backend_impl


class Backend(ABC):
    """Interface for loading and saving production data."""

    @abstractmethod
    def load(self, limit: int) -> pd.DataFrame | None:
        """Load production data from the database."""

    @abstractmethod
    def save(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save production data to the database."""

    @abstractmethod
    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Label production data using auto-generated ground truth values."""

    def get_fake_label(self, prediction, ground_truth_quality):
        """Generate a fake ground truth label for a sample.

        This function will randomly return a ground truth label taking into account the
        prediction quality we want to achieve.
        """
        return (
            prediction
            if random.random() < ground_truth_quality
            else random.choice(["Adelie", "Chinstrap", "Gentoo"])
        )


class SQLite(Backend):
    """SQLite implementation for loading and saving production data."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize backend using the supplied configuration.

        If the configuration is not provided, the class will attempt to read the
        configuration from environment variables.
        """
        self.database = "penguins.db"

        if config:
            self.database = config.get("database", self.database)
        else:
            self.database = os.getenv("MODEL_BACKEND_DATABASE", self.database)

        logging.info("Backend database: %s", self.database)

    def load(self, limit: int = 100) -> pd.DataFrame | None:
        """Load production data from a SQLite database."""
        import pandas as pd

        if not Path(self.database).exists():
            logging.error("Database %s does not exist.", self.database)
            return None

        connection = sqlite3.connect(self.database)

        query = (
            "SELECT island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, "
            "body_mass_g, prediction, ground_truth FROM data "
            "ORDER BY date DESC LIMIT ?;"
        )

        data = pd.read_sql_query(query, connection, params=(limit,))
        connection.close()

        return data

    def save(self, model_input: pd.DataFrame, model_output: list):
        """Save production data to a SQLite database.

        If the database doesn't exist, this function will create it.
        """
        logging.info("Storing production data in the database...")

        connection = None
        try:
            connection = sqlite3.connect(self.database)

            # Let's create a copy from the model input so we can modify the DataFrame
            # before storing it in the database.
            data = model_input.copy()

            # We need to add the current date and time so we can filter data based on
            # when it was collected.
            data["date"] = datetime.now(timezone.utc)

            # Let's initialize the prediction and confidence columns with None. We'll
            # overwrite them later if the model output is not empty.
            data["prediction"] = None
            data["confidence"] = None

            # Let's also add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual species for the data.
            data["ground_truth"] = None

            # If the model output is not empty, we should update the prediction and
            # confidence columns with the corresponding values.
            if model_output is not None and len(model_output) > 0:
                data["prediction"] = [item["prediction"] for item in model_output]
                data["confidence"] = [item["confidence"] for item in model_output]

            # Let's automatically generate a unique identified for each row in the
            # DataFrame. This will be helpful later when labeling the data.
            data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

            # Finally, we can save the data to the database.
            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error:
            logging.exception(
                "There was an error saving production data to the database.",
            )
        finally:
            if connection:
                connection.close()

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Label production data using auto-generated ground truth values."""
        if not Path(self.database).exists():
            logging.error("Database %s does not exist.", self.database)
            return 0

        connection = None
        try:
            connection = sqlite3.connect(self.database)

            # We want to return any unlabeled samples from the database.
            df = pd.read_sql_query(
                "SELECT * FROM data WHERE ground_truth IS NULL",
                connection,
            )
            logging.info("Loaded %s unlabeled samples.", len(df))

            # If there are no unlabeled samples, we don't need to do anything else.
            if df.empty:
                return 0

            for _, row in df.iterrows():
                uuid = row["uuid"]
                label = self.get_fake_label(row["prediction"], ground_truth_quality)

                # Update the database
                update_query = "UPDATE data SET ground_truth = ? WHERE uuid = ?"
                connection.execute(update_query, (label, uuid))

            connection.commit()
            return len(df)
        except Exception:
            logging.exception("There was an error labeling production data")
            return 0
        finally:
            if connection:
                connection.close()


class S3(Backend):
    """S3 implementation for loading and saving production data."""

    def __init__(self) -> None:
        """Initialize the S3 bucket and key."""

        # location of data
        # location of ground truth
        # assume role

    def load(self, limit: int = 100) -> pd.DataFrame:
        """Load production data from an S3 bucket."""
        return None
