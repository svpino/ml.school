# Foreach branches

Metaflow supports creating a dynamic number of parallel branches using a [foreach](https://docs.metaflow.org/metaflow/basics#foreach) branch. 

You can spin up multiple branches as part of a `self.next()` call by using the `foreach` keyword together with the name of a list. Metaflow will create separate tasks to process each item from the list.

In the example code, Metaflow creates one task for each name in the list `people`. The `capitalize` step will run with every person from the list as an input. Notice how you can access the specific name assigned to the task using the `self.input` attribute.

![Foreach branches](.guide/introduction-to-metaflow/images/foreach.png)

Finally, you must join foreach branches just like static branches. The example code uses the `inputs` parameter to iterate over each person's name.

Run the following command in the terminal to execute the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/foreach.py run
```


