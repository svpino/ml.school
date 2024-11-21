# Metaflow artifacts

Metaflow uses [artifacts](https://docs.metaflow.org/metaflow/basics#artifacts) to track all the values assigned to an instance variable within a flow.

Artifacts simplify managing the flow of data through a workflow. Metaflow automatically persists artifacts and carries their value from one step to another. You can access the value of these artifacts at any point for debugging and analysis purposes.

Artifacts behave consistently across different environments, whether you run steps locally or in the cloud. 

In the example code, `self.variable` is an artifact. The flow initializes it in the `start` step, increments it in the `increment` step, and finally prints it in the `end` step.

If you decide to run the `increment` step in the cloud and the `end` step locally, Metaflow would ensure that the artifact's updated value moves through the steps correctly.


