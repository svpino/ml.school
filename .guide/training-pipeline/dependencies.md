# Pipeline Dependencies

As we build the pipeline, we'll need to specify the list of packages necessary to execute the code. We'll want Metaflow to download specific versions for some of these packages and the latest available version for others.

We'll use the `@conda_base` and `@conda` Metaflow decorators to specify these dependencies in the flow. To avoid unnecessary code duplication, we'll create a simple function named `packages()` to manage the list of dependencies.

This function will take a list of package names and return them in the appropriate format, including the version required by the pipeline. We'll keep the list of versioned packages centralized in a `PACKAGES` dictionary.

We'll use this function across different pipelines, so we want to add it to [`common.py`](pipelines/common.py), a Python file we can import everywhere. This file contains shared code that we can reuse across different files.

You can run the [tests](tests/test_common_packages.py) associated with the `packages()` function by executing the following command:

```shell
uv run -- pytest -k test_common_packages
```