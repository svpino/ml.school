import importlib
import json
import logging
import os
import random
import re
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
            backend_impl = getattr(module, cls)(config=self._get_config())
        except Exception as e:
            message = f"There was an error instantiating class {self.backend}."
            raise RuntimeError(message) from e
        else:
            logging.info("Backend: %s", self.backend)
            return backend_impl

    def _get_config(self):
        """Return the endpoint configuration with environment variables expanded."""
        if not self.backend_config:
            return None

        config = self.backend_config.to_dict()
        pattern = re.compile(r"\$\{(\w+)\}")

        def replacer(match):
            env_var = match.group(1)
            return os.getenv(env_var, f"${{{env_var}}}")

        for key, value in self.backend_config.items():
            if isinstance(value, str):
                config[key] = pattern.sub(replacer, value)

        return config


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

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the S3 bucket and key."""
        # location of data
        # location of ground truth
        # assume role
        self.data_capture_destination = (
            config.get("data-capture-destination", None) if config else None
        )
        self.assume_role = config.get("assume_role", None) if config else None
        self.ground_truth_uri = config.get("ground-truth-uri", None) if config else None

        # Let's make sure the ground truth uri ends with a '/'
        self.ground_truth_uri = self.ground_truth_uri.rstrip("/") + "/"

        logging.info("Data capture destination: %s", self.data_capture_destination)
        logging.info("Ground truth URI: %s", self.ground_truth_uri)
        logging.info("Assume role: %s", self.assume_role)

    def load(self, limit: int = 100) -> pd.DataFrame:
        """Load production data from an S3 bucket."""
        data = self._load_labeled_data(
            data_uri=self.data_capture_destination,
            ground_truth_uri=self.ground_truth_uri,
        )

        # We need to remove a few columns that are not needed for the monitoring tests.
        return data.drop(columns=["date", "event_id", "confidence"])

    def label(self, ground_truth_quality=0.8):
        """Generate ground truth labels for data captured by a SageMaker endpoint.

        This function loads any unlabeled data from the location where SageMaker stores
        the data captured by the endpoint and generates fake ground truth labels. The
        function stores the labels in the specified S3 location.
        """
        import json
        from datetime import datetime, timezone

        import boto3

        ground_truth_uri = self.ground_truth_uri.rstrip("/") + "/"

        s3 = boto3.client("s3")
        data = self._load_unlabeled_data(s3)

        logging.info("Loaded %s unlabeled samples from S3.", len(data))

        # If there are no unlabeled samples, we don't need to do anything else.
        if data.empty:
            return 0

        records = []
        for event_id, group in data.groupby("event_id"):
            predictions = []
            for _, row in group.iterrows():
                predictions.append(
                    self.get_fake_label(row["prediction"], ground_truth_quality),
                )

            record = {
                "groundTruthData": {
                    # For testing purposes, we will generate a random
                    # label for each request.
                    "data": predictions,
                    "encoding": "CSV",
                },
                "eventMetadata": {
                    # This value should match the id of the request
                    # captured by the endpoint.
                    "eventId": event_id,
                },
                "eventVersion": "0",
            }

            records.append(json.dumps(record))

        ground_truth_payload = "\n".join(records)
        upload_time = datetime.now(tz=timezone.utc)
        uri = (
            "/".join(ground_truth_uri.split("/")[3:])
            + f"{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
        )

        s3.put_object(
            Body=ground_truth_payload,
            Bucket=ground_truth_uri.split("/")[2],
            Key=uri,
        )

        return len(data)

    def _load_unlabeled_data(self, s3):
        """Load any unlabeled data from the specified S3 location.

        This function will load the data captured from the endpoint during inference that
        does not have a corresponding ground truth information.
        """
        data = self._load_collected_data(s3)
        return data if data.empty else data[data["species"].isna()]

    def _load_collected_data(self, s3):
        """Load the data capture from the endpoint and merge it with its ground truth."""
        data = self._load_collected_data_files(s3)
        ground_truth = self._load_ground_truth_files(s3)

        if len(data) == 0:
            return pd.DataFrame()

        if len(ground_truth) > 0:
            ground_truth = ground_truth.explode("species")
            data["index"] = data.groupby("event_id").cumcount()
            ground_truth["index"] = ground_truth.groupby("event_id").cumcount()

            data = data.merge(
                ground_truth,
                on=["event_id", "index"],
                how="left",
            )
            data = data.rename(columns={"species_y": "species"}).drop(
                columns=["species_x", "index"],
            )

        return data

    def _load_ground_truth_files(self, s3):
        """Load the ground truth data from the specified S3 location."""

        def process(row):
            data = row["groundTruthData"]["data"]
            event_id = row["eventMetadata"]["eventId"]

            return pd.DataFrame({"event_id": [event_id], "species": [data]})

        df = self._load_files(s3, self.ground_truth_uri)

        if df is None:
            return pd.DataFrame()

        processed_dfs = [process(row) for _, row in df.iterrows()]

        return pd.concat(processed_dfs, ignore_index=True)

    def _load_collected_data_files(self, s3):
        """Load the data captured from the endpoint during inference."""

        def process_row(row):
            date = row["eventMetadata"]["inferenceTime"]
            event_id = row["eventMetadata"]["eventId"]
            input_data = json.loads(row["captureData"]["endpointInput"]["data"])
            output_data = json.loads(row["captureData"]["endpointOutput"]["data"])

            if "instances" in input_data:
                df = pd.DataFrame(input_data["instances"])
            elif "inputs" in input_data:
                df = pd.DataFrame(input_data["inputs"])
            else:
                df = pd.DataFrame(
                    input_data["dataframe_split"]["data"],
                    columns=input_data["dataframe_split"]["columns"],
                )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(output_data["predictions"]),
                ],
                axis=1,
            )

            df["date"] = date
            df["event_id"] = event_id
            df["species"] = None
            return df

        df = self._load_files(s3, self.data_capture_destination)

        if df is None:
            return pd.DataFrame()

        processed_dfs = [process_row(row) for _, row in df.iterrows()]

        # Concatenate all processed DataFrames
        result_df = pd.concat(processed_dfs, ignore_index=True)
        return result_df.sort_values(by="date", ascending=False).reset_index(drop=True)

    def _load_files(self, s3, s3_uri):
        """Load every file stored in the supplied S3 location.

        This function will recursively return the contents of every file stored under the
        specified location. The function assumes that the files are stored in JSON Lines
        format.
        """
        bucket = s3_uri.split("/")[2]
        prefix = "/".join(s3_uri.split("/")[3:])

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        files = [
            obj["Key"]
            for page in pages
            if "Contents" in page
            for obj in page["Contents"]
        ]

        if len(files) == 0:
            return None

        dfs = []
        for file in files:
            obj = s3.get_object(Bucket=bucket, Key=file)
            data = obj["Body"].read().decode("utf-8")

            json_lines = data.splitlines()

            # Parse each line as a JSON object and collect into a list
            dfs.append(pd.DataFrame([json.loads(line) for line in json_lines]))

        # Concatenate all DataFrames into a single DataFrame
        return pd.concat(dfs, ignore_index=True)

    def save(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Not implemented."""


class Mock(Backend):
    """Mock implementation of the Backend abstract class.

    This class is helpful for testing purposes to simulate access to
    a production backend.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize the mock backend."""

    def load(self, limit: int) -> pd.DataFrame | None:  # noqa: ARG002
        """Return fake data for testing purposes."""
        return pd.DataFrame(
            [
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 38.6,
                    "culmen_depth_mm": 21.2,
                    "flipper_length_mm": 191,
                    "body_mass_g": 3800,
                    "sex": "MALE",
                    "ground_truth": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 34.6,
                    "culmen_depth_mm": 21.1,
                    "flipper_length_mm": 198,
                    "body_mass_g": 4400,
                    "sex": "MALE",
                    "ground_truth": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 36.6,
                    "culmen_depth_mm": 17.8,
                    "flipper_length_mm": 185,
                    "body_mass_g": 3700,
                    "sex": "FEMALE",
                    "ground_truth": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 38.7,
                    "culmen_depth_mm": 19,
                    "flipper_length_mm": 195,
                    "body_mass_g": 3450,
                    "sex": "FEMALE",
                    "ground_truth": "Adelie",
                    "prediction": "Adelie",
                },
                {
                    "island": "Torgersen",
                    "culmen_length_mm": 42.5,
                    "culmen_depth_mm": 20.7,
                    "flipper_length_mm": 197,
                    "body_mass_g": 4500,
                    "sex": "MALE",
                    "ground_truth": "Adelie",
                    "prediction": "Adelie",
                },
            ],
        )

    def save(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Not implemented."""

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Not implemented."""
