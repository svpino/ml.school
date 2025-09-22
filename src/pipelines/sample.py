from metaflow import step

from common.pipeline import Pipeline


class Sample(Pipeline):
    @step
    def start(self):
        import mlflow

        run_id = "c5b4a7d376974936af0f67ab5fac08e8"

        mlflow.pyfunc.load_model(
            model_uri="models:/m-5ec93002a0794860a2c2c8ed766b1469",
            dst_path="/tmp/model",
        )

        self.next(self.process)

    @step
    def process(self):
        self.next(self.end)

    @step
    def end(self):
        print("Done")


if __name__ == "__main__":
    Sample()
