
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3


def load_data_from_s3(location: str):
    """Load the dataset from an S3 location.

    This function will concatenate every CSV file in the given location
    and return a single DataFrame.
    """
    print(f"Loading dataset from location {location}")

    with S3(s3root=location) as s3:
        files = s3.get_all()

        print(f"Found {len(files)} file(s) in remote location")

        raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
        return pd.concat(raw_data)


def load_data_from_file(dataset_location):
    """Load the dataset from a local file.

    This function is useful to test the pipeline locally
    without having to access the data remotely.
    """
    location = Path(dataset_location)
    print(f"Loading dataset from location {location.as_posix()}")
    return pd.read_csv(location)


def load_data(dataset_location, debug=False):
    if debug:
        df = load_data_from_file(dataset_location)
    else:
        df = load_data_from_s3(dataset_location)

    # Shuffle the data
    data = df.sample(frac=1, random_state=42)

    print(f"Loaded dataset with {len(data)} samples")

    return data
