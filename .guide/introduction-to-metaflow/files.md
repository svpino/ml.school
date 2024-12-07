# Including Files

In Metaflow, you can read the contents of a local file and [include](https://docs.metaflow.org/scaling/data#data-in-local-files) it as part of a flow. Metaflow will version these files and make them accessible to all the steps in the flow.

The example code includes a text file named `file` as part of the flow. If specified, Metaflow will make the file's contents available at every step of the flow.

Specifying a file when running a flow is similar to specifying a Metaflow parameter:

```bash
uv run -- python .guide/introduction-to-metaflow/files.py run \
    --file .guide/introduction-to-metaflow/sample.csv
```

You can also initialize a file with a default value that will be used in case the file is not specified when running the flow:

```python
file = IncludeFile(
    "file",
    is_text=True,
    help="Sample comma-separated file",
    default="path/to/sample.csv"
)
```