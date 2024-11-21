# Metaflow foreach

Metaflow supports creating a dynamic number of parallel branches using a [foreach](https://docs.metaflow.org/metaflow/basics#foreach) branch. 

You can spin up multiple branches using the `foreach` keyword to specify the name of a list as part of the `self.next()` function. Metaflow will create separate tasks to process each item from the list.

In the example code, Metaflow creates one task for each name in the list `people` and calls the `capitalize` step. Notice how you can access the specific name assigned to the task using `self.input`.

Finally, you must join foreach branches just like static branches. The example code uses the `inputs` parameter to iterate over each person's name.

