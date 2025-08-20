import pytest
from metaflow import Run, Runner

from pipelines.inference import backend


@pytest.fixture(scope="module")
def metaflow_data():
    with Runner("tests/mock_pipeline.py", show_output=True).run() as running:
        return Run(running.run.pathspec).data


def test_dataset_decorator_loads_data(metaflow_data):
    assert len(metaflow_data.data) > 0


def test_backend_decorator_loads_backend(metaflow_data):
    assert isinstance(metaflow_data.backend_impl, backend.Local)


def test_dataset_decorator_cleans_sex_column(metaflow_data):
    sex_distribution = metaflow_data.data["sex"].value_counts()

    assert len(sex_distribution) == len(["MALE", "FEMALE"])
    assert sex_distribution.index[0] == "MALE"
    assert sex_distribution.index[1] == "FEMALE"
