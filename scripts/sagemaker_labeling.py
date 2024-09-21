import sys
from pathlib import Path

import boto3

sys.path.append(str(Path(__file__).parent.parent))

from pipelines.common import SagemakerLabeling

if __name__ == "__main__":
    s3_client = boto3.client("s3")
    labeling = SagemakerLabeling(
        s3_client=s3_client,
        data_capture_s3_uri="s3://mlschool/penguins/monitoring/data-capture/",
        ground_truth_s3_uri="s3://mlschool/penguins/monitoring/groundtruth/",
    )

    labeling.label()
    print(labeling.load_labeled_data())
