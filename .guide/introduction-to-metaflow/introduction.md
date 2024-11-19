# Introduction to Metaflow

[Metaflow](https://metaflow.org) is an open-source Python library originally developed at Netflix to simplify the development, deployment, and management of data science, machine learning, and artificial intelligence applications. 

Metaflow provides a consistent framework for handling components such as data storage, compute resources, orchestration, version control, deployment, and modeling. It integrates with every major cloud platform and Kubernetes.

Check [What is Metaflow](https://docs.metaflow.org/introduction/what-is-metaflow) for more information about the library and how you can use it.

Metaflow models a program as a directed graph of operations called a **flow**. Each operation—or **step**—in the flow is a node in the graph and contains transitions to other steps.

Every flow must extend the `FlowSpec` class and implement a `start` and `end` step. Every step is denoted with the `@step` decorator. From a step, you can use the `self.next()` method to transition to the next step.

You can execute a flow by running the Python file followed by the `run` command:

```bash
python flow.py run
```

The example code defines a basic, linear flow with four steps named `HelloWorld`.

<div class="video">
    <iframe
        src="https://www.youtube.com/embed/vSrq88BlzIE?si=TpTYdjWzIGnbobDd"
        title="YouTube video player"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
    ></iframe>
</div>
