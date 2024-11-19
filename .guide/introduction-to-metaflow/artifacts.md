# Metaflow artifacts

Metaflow keeps track of all the values assigned to an instance variable within a flow. These are called "artifacts," and they are a core concept in Metaflow.

Artifacts simplify managing the flow of data through a workflow. Metaflow automatically persists artifacts and carries their value from one step to another. You can access the value of these artifacts at any point for debugging and analysis purposes.

Artifacts behave consistently across different environments, regardless of whether you run steps locally or in the cloud. 

In the example code, the variable `self.variable` is an artifact. The flow initializes it in the `start` step, increments it in the `increment` step, and finally prints it in the `end` step.

If you decide to run the `increment` step in the cloud and the `end` step locally, Metaflow would ensure that the updated value of the artifact moves through the steps correctly.

The `Artifacts` flow works as follows:

1. It initializes `variable` to `1` in the `start` step.
2. It increments `variable` by `2` in the `increment` step.
3. It prints the final value of `variable` in the `end` step.

Executing the flow should print `3` as the final value of the artifact `variable`.
