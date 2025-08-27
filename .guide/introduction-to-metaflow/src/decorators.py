from metaflow import FlowMutator, FlowSpec, step, user_step_decorator


@user_step_decorator
def log(step_name, flow, inputs=None, attributes=None):  # noqa: ARG001
    """Log when a step starts and ends."""
    print(f"Starting step: {step_name}")
    yield
    print(f"Finished step: {step_name}")


class logger_flow(FlowMutator):  # noqa: N801
    """Apply logging decorator to all steps in a flow."""

    def mutate(self, mutable_flow):
        """Add the log decorator to every step in the flow."""
        for _, flow_step in mutable_flow.steps:
            flow_step.add_decorator("log", duplicates=flow_step.IGNORE)


@logger_flow
class Decorators(FlowSpec):
    """A flow that demonstrates how to use custom decorators and mutators."""

    @step
    def start(self):
        """Initialize the flow with some data."""
        self.message = "Hello from Metaflow!"
        print(f"Message: {self.message}")
        self.next(self.process)

    @step
    def process(self):
        """Process the message."""
        self.message = self.message.upper()
        self.next(self.end)

    @step
    def end(self):
        """Finalize and print the results."""
        print(f"Message: {self.message}")


if __name__ == "__main__":
    Decorators()
