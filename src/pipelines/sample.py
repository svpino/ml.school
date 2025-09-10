from metaflow import FlowSpec, step


class Sample(FlowSpec):
    @step
    def start(self):
        self.matrix = [[0] * 100 for _ in range(100)]
        self.next(self.process)

    @step
    def process(self):
        for i in range(100):
            for j in range(100):
                self.matrix[i][j] = 1

        self.next(self.end)

    @step
    def end(self):
        print("Done")


if __name__ == "__main__":
    Sample()
