import importlib
import os
import re
import sys
import time
from contextlib import suppress
from pathlib import Path

import pandas as pd
import yaml
from metaflow import (
    Config,
    FlowMutator,
    FlowSpec,
    Parameter,
    config_expr,
    current,
    project,
    user_step_decorator,
)


@user_step_decorator
def dataset(step_name, flow, inputs=None, attr=None):  # noqa: ARG001
    """Load and prepare the dataset.

    This decorator loads the dataset, replaces some extraneous values
    with NaN, and shuffles the data before creating an artifact on the
    current flow.
    """
    import numpy as np

    # Let's check if the dataset file exists
    if not Path(flow.dataset).exists():
        # If we don't find the dataset file, we can set the artifact to None
        # and let the step continue.
        flow.data = None
        yield
    else:
        # If we find the dataset file, we can load it and process it.
        data = pd.read_csv(flow.dataset)

        # Replace extraneous values in the sex column with NaN
        data["sex"] = data["sex"].replace(".", np.nan)

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        flow.logger.info("Loaded dataset with %d samples", len(data))

        # Let's now create an artifact on the current flow so every step of the flow
        # has access to it.
        flow.data = data
        yield


@user_step_decorator
def logging(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Configure the logging handler.

    We need to configure the logging handler on every individual step of a pipeline.
    This decorator will do that, and will set an artifact so every step in the flow
    has access to it.
    """
    import logging
    import logging.config

    # Let's get the logging configuration file from the project settings.
    logging_file = flow.project.get("logging", "logging.conf")

    if Path(logging_file).exists():
        logging.config.fileConfig(logging_file)
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

    flow.logger = logging.getLogger("mlschool")
    yield


@user_step_decorator
def mlflow(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Configure MLflow's tracking URI for the current step."""
    import mlflow

    mlflow.set_tracking_uri(flow.mlflow_tracking_uri)
    yield


@user_step_decorator
def backend(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Instantiate the backend implementation.

    This decorator will load the backend configuration and use it to create
    an instance of the backend implementation class. We'll create an artifact
    so every step in the flow has access to it.
    """
    # For the configuration to remain clean and easy to remember, we want to
    # reference backend classes as "backend.<class_name>" without having to include
    # their full class path. To accomplish this, we need to import the
    # inference.backend module so it's available to the `import_module` call.
    with suppress(ImportError):
        import inference.backend  # noqa: F401

    try:
        # Let's import the module containing the backend implementation.
        module, cls = flow.backend.rsplit(".", 1)
        module = importlib.import_module(module)

        # Now, we can instantiate the class using the backend configuration
        # settings coming from the configuration file.
        backend_impl = getattr(module, cls)(config=flow.project.backend)
    except Exception as e:
        message = f"There was an error instantiating class {flow.backend}"
        flow.logger.exception(message)
        raise RuntimeError(message) from e
    else:
        flow.logger.info("Backend: %s", flow.backend)
        flow.backend_impl = backend_impl
        yield


def parse_project_configuration(x):
    """Parse the project configuration from the supplied configuration file.

    This function will expand any environment variables that are present in the
    backend configuration values. The environment variables should be in the format
    `${ENVIRONMENT_VARIABLE}`.
    """
    config = yaml.full_load(x)

    # If the tracking URI is not part of the configuration, we want to use
    # the value of the `MLFLOW_TRACKING_URI` environment variable.
    if "mlflow_tracking_uri" not in config:
        config["mlflow_tracking_uri"] = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

    if "backend" not in config:
        config["backend"] = {"module": "backend.Local"}

    # This regex matches any environment variable in the format ${ENVIRONMENT_VARIABLE}
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"${{{env_var}}}")

    for key, value in config["backend"].items():
        if isinstance(value, str):
            config["backend"][key] = pattern.sub(replacer, value)

    return config


class pipeline(FlowMutator):  # noqa: N801
    """Mutate a flow by applying a set of decorators to every step."""

    def mutate(self, mutable_flow):
        """Mutates the supplied flow."""
        for _, step in mutable_flow.steps:
            # We want every step to have access to a preconfigured logger.
            step.add_decorator("logging", duplicates=step.IGNORE)

            # We want to configure the MLflow tracking URI on every step.
            step.add_decorator("mlflow", duplicates=step.IGNORE)


@pipeline
@project(name=config_expr("project.project"))
class Pipeline(FlowSpec):
    """Foundation flow for pipelines that require access to the dataset and backend."""

    project = Config(
        "project",
        help="Project configuration settings.",
        default="config/local.yml",
        parser=parse_project_configuration,
    )

    backend = Parameter(
        "backend",
        help="Backend module implementation.",
        default=project.backend["module"],
    )

    dataset = Parameter(
        "dataset",
        help="Project dataset that will be used to train and evaluate the model.",
        default="data/penguins.csv",
    )

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="MLflow tracking URI.",
        default=project.mlflow_tracking_uri,
    )
