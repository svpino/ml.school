from metaflow import FlowSpec, step


class Branches(FlowSpec):
    """A flow that showcases how branches work."""

    @step
    def start(self):
        """Initialize the start value artifact."""
        self.start_value = 1
        self.next(self.step1, self.step2)

    @step
    def step1(self):
        """Assign a value to a new artifact."""
        print("Executing Step 1")
        self.value = 1
        self.next(self.join)

    @step
    def step2(self):
        """Assign a value to a new artifact."""
        print("Executing Step 2")
        self.value = 2
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the two branches."""
        print("Executing the Join Step")

        self.merge_artifacts(inputs, include=["start_value"])

        print("Step 1's value:", inputs.step1.value)
        print("Step 2's value:", inputs.step2.value)

        self.final_value = sum(i.value for i in inputs)
        self.next(self.end)

    @step
    def end(self):
        """Print the final artifact values."""
        print("Start value:", self.start_value)
        print("Final value:", self.final_value)


if __name__ == "__main__":
    Branches()
