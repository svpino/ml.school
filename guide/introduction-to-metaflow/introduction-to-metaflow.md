# Introduction to Metaflow

[Metaflow](https://metaflow.org) is an open-source Python library originally developed at Netflix to simplify the development, deployment, and management of data science, machine learning, and artificial intelligence applications. 

Metaflow provides a consistent framework for handling data storage, computing resources, orchestration, version control, deployment, and modeling. It integrates with every major cloud platform and Kubernetes.

Check the page ["What is Metaflow"](https://docs.metaflow.org/introduction/what-is-metaflow) for more information about the library and how you can use it.

Metaflow models a program as a directed graph of operations called a **flow**. Each operation—or **step**—in the flow is a node in the graph and contains transitions to other steps.

Every flow must extend the `FlowSpec` class and implement a `start` and `end` step. Every step is denoted with the `@step` decorator. From a step, you can use the `self.next()` method to transition to the next step.

The example code accompanying this section defines a basic, linear flow with four steps. Each step prints a message and goes to the next one.

![Linear flow](.guide/introduction-to-metaflow/images/linear.png)

You can run the flow by executing the following command:

```shell
uv run .guide/introduction-to-metaflow/src/introduction.py run
```

This command will use Metaflow to run the flow defined in the [`introduction.py`](.guide/introduction-to-metaflow/src/introduction.py) file. 
