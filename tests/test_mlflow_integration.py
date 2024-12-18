import pytest
from metaflow import Runner

# @pytest.fixture(scope="module")
# def metaflow_data():
#     temporal_directory = tempfile.gettempdir()
#     mlflow_directory = Path(temporal_directory) / "mlflow"

#     cmd = [
#         "python",
#         "pipelines/training.py",
#         "--environment=conda",
#         "run",
#         "--mlflow-tracking-uri",
#         mlflow_directory.as_posix(),
#         "--run-id-file",
#         "test_id",
#     ]
#     subprocess.check_call(cmd)
#     with open("test_id") as f:
#         run = Flow("Training")[f.read()]
#         return run.data


def test_sample(metaflow_data):
    assert metaflow_data.mlflow_run_id is not None


@pytest.fixture(scope="module")
def metaflow_data():
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=True,
    ).run() as running:
        return running.run.data
