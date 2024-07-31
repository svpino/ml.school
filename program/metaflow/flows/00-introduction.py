from metaflow import FlowSpec, step


class TrainingFlow(FlowSpec):
    @step
    def start(self):
        print("Start")
        self.next(self.end)

    @step
    def end(self):
        print("the end")


if __name__ == "__main__":
    TrainingFlow()
