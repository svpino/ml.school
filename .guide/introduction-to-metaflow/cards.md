# Visualizing Results

Metaflow uses [cards](https://docs.metaflow.org/metaflow/visualizing-results) to generate and display reports during a flow's execution. Cards are customizable HTML files with images, text, and tables to help analyze task results.

You can use Metaflow cards in many different scenarios, such as reporting model performance after training and evaluating a model, monitoring the progress of long-running tasks, sharing human-readable results with stakeholders, tracking experiments, and comparing results across multiple runs.

You can attach a card to any existing step, and Metaflow will generate a default card with all artifacts produced by the task. You can also create a custom card using built-in components or HTML content. Once Metaflow completes the step, it will run additional code to generate the attached card.

In the example code, the `start` and `end` steps use the default Metaflow card, and the `report` step uses a custom HTML card.

Run the following command in the terminal to execute the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/cards.py run
```

After running a flow, you can visualize the card attached to a step by opening it on the command line. The following command will open the card attached to the `report` step in your default web browser: 

```shell
uv run -- python .guide/introduction-to-metaflow/cards.py card view report
```

Metaflow provides a [built-in card viewer](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards#using-local-card-viewer) that sets up a local server for viewing cards. Open a new terminal, navigate to the same working directory where you are executing the flow, and run the following command:

```bash
uv run -- python .guide/introduction-to-metaflow/cards.py card server
```

After this, open [localhost:8324](http://localhost:8324) to see a page with every card produced by the latest run of the flow.
