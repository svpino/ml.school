from metaflow import FlowSpec, step


class Branches(FlowSpec):
    """A flow that showcases how branches work."""

    @step
    def start(self):
        """Initialize the start value artifact."""
        self.start_value = 0
        self.next(self.step1, self.step2)

    @step
    def step1(self):
        """Assign a value to an artifact."""
        print("Executing Step 1")
        self.common = 1
        self.next(self.join)

    @step
    def step2(self):
        """Assign a value to an artifact."""
        print("Executing Step 2")
        self.common = 2
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the two branches."""
        self.merge_artifacts(inputs, exclude=["common"])

        print("Step 1's artifact value:", inputs.step1.common)
        print("Step 2's artifact value:", inputs.step2.common)

        self.final_value = sum(i.common for i in inputs)
        self.next(self.end)

    @step
    def end(self):
        """Print the final artifact values."""
        print("Start value:", self.start_value)
        print("Final value:", self.final_value)


if __name__ == "__main__":
    Branches()
