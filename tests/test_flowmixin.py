import subprocess
import sys
from pathlib import Path

import pandas as pd
from metaflow.client.core import Flow


def test_load_dataset():
    penguins = pd.read_csv(Path("data/penguins.csv"))
    run = run_pipeline()
    assert len(run.data.data) == len(penguins)


def test_load_dataset_cleans_sex_column():
    run = run_pipeline()

    sex_distribution = run.data.data["sex"].value_counts()

    assert len(sex_distribution) == 2
    assert sex_distribution.index[0] == "MALE"
    assert sex_distribution.index[1] == "FEMALE"


def run_pipeline():
    run_id_file = "tests/.flowmixin"
    subprocess.check_call(  # noqa: S603
        [
            sys.executable,
            "tests/flowmixin_flow.py",
            "run",
            "--run-id-file",
            run_id_file,
        ],
    )

    with Path(run_id_file).open() as f:
        return Flow("TestFlowMixinFlow")[f.read()]
