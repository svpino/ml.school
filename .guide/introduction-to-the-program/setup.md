# Preparing Your Environment

Navigate to the project's root directory and create and activate a virtual environment. We'll install all the required libraries inside this environment:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Within the virtual environment, update `pip` and install the libraries listed in the `requirements.txt` file:

```shell
pip3 install -U pip && pip3 install -r requirements.txt
```

At this point, you should have a working Python environment with all the required dependencies. You can ensure everything is working correctly by checking the version of one of the installed libraries:

```shell
pip3 show metaflow
```

