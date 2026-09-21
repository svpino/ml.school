# Decorators and Mutators

Metaflow allows you to extend its functionality by creating [custom decorators](https://docs.metaflow.org/metaflow/composing-flows/custom-decorators) and [mutators](https://docs.metaflow.org/metaflow/composing-flows/mutators). Custom decorators enable you to add pre and post-processing logic to individual steps, while mutators allow you to programmatically modify flows by applying decorators to multiple steps at once.

## Custom Decorators

A custom decorator is a function that can execute code before and after a step runs. You create custom decorators using the `@user_step_decorator` decorator. The decorator function executes code before the `yield` statement (pre-processing), allows the step to run during `yield`, and then executes code after `yield` (post-processing):

```python
@user_step_decorator
def log(step_name, flow, inputs=None, attributes=None):
    """Log when a step starts and ends."""
    print(f"Starting step: {step_name}")
    yield
    print(f"Finished step: {step_name}")
```

You can apply this custom decorator to any step by adding `@log` above the `@step` decorator:

```python
@log
@step
def start(self):
    print("Hello from the start step")
    self.next(self.end)
```

## Mutators

A [mutator](https://docs.metaflow.org/metaflow/composing-flows/mutators) is a class that inherits from `FlowMutator` and can programmatically modify a flow before it runs. The `mutate` method receives a `mutable_flow` object and can add decorators to steps, modify parameters, or change flow structure.

```python
class logger_flow(FlowMutator):
    """Apply logging decorator to all steps in a flow."""
    
    def mutate(self, mutable_flow):
        for _, step in mutable_flow.steps:
            step.add_decorator("log", duplicates=step.IGNORE)
```

You apply a mutator to a flow class using the mutator name as a decorator: `@logger_flow`.

The example code demonstrates a flow with a custom logging decorator that is automatically applied to all steps via a mutator. Each step will show start and end messages, making it easy to track execution progress.

You can run the following command to execute the flow:

```shell
uv run .guide/introduction-to-metaflow/src/decorators.py run
```
