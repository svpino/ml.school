# Sketching the Workflow

The first step when building a pipeline is creating the overall structure of the workflow. The goal is to assemble a skeleton with the basic building blocks of the pipeline and have it run as soon as possible.

This step is similar to an artist sketching an initial rough draft of a painting before anything else. At this stage, the details don't matter. You want to focus on capturing the broad strokes of the pipeline.

As part of the Training pipeline, we'll use a cross-validation process to build the model, so we'll need steps to transform, train, and evaluate the individual models we'll create for each one of the cross-validation folds.

We'll use the entire dataset to train the final model, so we'll need a branch to handle this process as well. Finally, we'll need a step to register the final model in the model registry.

![Training pipeline](.guide/training-pipeline/images/training.png)

After we have the main structure in place, we can ensure the pipeline runs correctly and then focus on filling in the blanks.