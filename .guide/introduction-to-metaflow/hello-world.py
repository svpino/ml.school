from metaflow import FlowSpec, step


class HelloWorld(FlowSpec):
    """A basic, linear flow with four steps.

    Every Metaflow flow must extend the `FlowSpec` class and implement a `start` and
    `end` step.
    """

    @step
    def start(self):
        """Every flow must start with a 'start' step."""
        print("Starting the flow")
        self.next(self.step_a)

    @step
    def step_a(self):
        """Follows the 'start' step."""
        print("Step A")
        self.next(self.step_b)

    @step
    def step_b(self):
        """Follows Step A."""
        print("Step B")
        self.next(self.end)

    @step
    def end(self):
        """Every flow must end with an 'end' step."""
        print("Ending the flow")


if __name__ == "__main__":
    HelloWorld()