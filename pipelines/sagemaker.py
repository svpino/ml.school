import json

import pandas as pd


def load_labeled_data(s3_client, data_uri, ground_truth_uri):
    data = _load_collected_data(s3_client, data_uri, ground_truth_uri)
    return data[data["species"].notna()]


def load_unlabeled_data(s3_client, data_uri, ground_truth_uri):
    data = _load_collected_data(s3_client, data_uri, ground_truth_uri)
    return data[data["species"].isna()]


def _load_collected_data(s3_client, data_uri, ground_truth_uri):
    data = _load_collected_data_files(s3_client, data_uri)
    ground_truth = _load_ground_truth_files(s3_client, ground_truth_uri)

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


def _load_ground_truth_files(s3_client, ground_truth_s3_uri):
    def process(row):
        data = row["groundTruthData"]["data"]
        event_id = row["eventMetadata"]["eventId"]

        return pd.DataFrame({"event_id": [event_id], "species": [data]})

    df = _load_files(s3_client, ground_truth_s3_uri)

    if df is None:
        return pd.DataFrame()

    processed_dfs = [process(row) for _, row in df.iterrows()]

    return pd.concat(processed_dfs, ignore_index=True)


def _load_collected_data_files(s3_client, data_uri):
    def process_row(row):
        date = row["eventMetadata"]["inferenceTime"]
        event_id = row["eventMetadata"]["eventId"]
        input_data = json.loads(row["captureData"]["endpointInput"]["data"])
        output_data = json.loads(row["captureData"]["endpointOutput"]["data"])

        df = pd.concat(
            [
                (
                    pd.DataFrame(input_data["inputs"])
                    if "inputs" in input_data
                    else pd.DataFrame(
                        input_data["dataframe_split"]["data"],
                        columns=input_data["dataframe_split"]["columns"],
                    )
                ),
                pd.DataFrame(output_data["predictions"]),
            ],
            axis=1,
        )

        df["date"] = date
        df["event_id"] = event_id
        df["species"] = None
        return df

    df = _load_files(s3_client, data_uri)

    if df is None:
        return pd.DataFrame()

    # Process each row and collect results
    processed_dfs = [process_row(row) for _, row in df.iterrows()]

    # Concatenate all processed DataFrames
    result_df = pd.concat(processed_dfs, ignore_index=True)
    return result_df.sort_values(by="date", ascending=False).reset_index(drop=True)


def _load_files(s3_client, s3_uri):
    bucket = s3_uri.split("/")[2]
    prefix = "/".join(s3_uri.split("/")[3:])

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = [
        obj["Key"] for page in pages if "Contents" in page for obj in page["Contents"]
    ]

    if len(files) == 0:
        return None

    dfs = []
    for file in files:
        obj = s3_client.get_object(Bucket=bucket, Key=file)
        data = obj["Body"].read().decode("utf-8")

        json_lines = data.splitlines()

        # Parse each line as a JSON object and collect into a list
        dfs.append(pd.DataFrame([json.loads(line) for line in json_lines]))

    # Concatenate all DataFrames into a single DataFrame
    return pd.concat(dfs, ignore_index=True)
