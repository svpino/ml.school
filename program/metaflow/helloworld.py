from metaflow import S3, FlowSpec, Parameter, step


class LinearFlow(FlowSpec):
    dataset_location = Parameter(
        "dataset_location",
        help="Location to the initial dataset",
        default="s3://mlschool/penguins/data/",
    )

    @step
    def start(self):
        self.my_var = "hello world"
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load dataset from S3 location."""
        from io import StringIO

        import pandas as pd

        with S3(s3root=self.dataset_location) as s3:
            self.data = pd.read_csv(StringIO(s3.get("data.csv").text))

        self.next(self.a)

    @step
    def a(self):
        print(f"dataset len: {len(self.data)}")
        self.next(self.end)

    @step
    def end(self):
        print("the data artifact is still: %s" % self.my_var)


if __name__ == "__main__":
    LinearFlow()
