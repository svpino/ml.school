from pathlib import Path

import pandas as pd
from metaflow import Run, Runner


def test_load_dataset():
    penguins = pd.read_csv(Path("data/penguins.csv"))
    metaflow_data = run_pipeline()
    assert len(metaflow_data.data) == len(penguins)


def test_load_dataset_cleans_sex_column():
    metaflow_data = run_pipeline()

    sex_distribution = metaflow_data.data["sex"].value_counts()

    assert len(sex_distribution) == 2
    assert sex_distribution.index[0] == "MALE"
    assert sex_distribution.index[1] == "FEMALE"


def run_pipeline():
    with Runner("tests/flowmixin_flow.py", show_output=False).run() as running:
        return Run(running.run.pathspec).data
