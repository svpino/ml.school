# Preparing Your Environment

To setup and manage the project, we'll use [uv](https://github.com/astral-sh/uv), a modern Rust-based Python package and project manager tool that will considerably simplify working with the project.

To install uv in your system, follow the instructions in the [official documentation](https://docs.astral.sh/uv/). When you finish, check that you have the tool correctly installed by running the `uv` command:

```shell
uv --version
```

We'll be running many different commands throughout the project, and to simplify this process, we'll use [just](https://github.com/casey/just), a tool that will let us define recipes to automate common commands, making our workflow much more efficient.

Follow the steps in the [official documentation](https://github.com/casey/just) to install the tool. Once installed, ensure everything is set up correctly by running the `just` command:

```shell
just --version
```

We'll also use [Docker](https://www.docker.com/) to run and deploy the models we'll build as part of the project. To install Docker, follow the instructions corresponding to your operating system in the [Docker documentation](https://docs.docker.com/engine/install/). After installation, confirm that Docker is working properly by running the following command:

```shell
docker ps
```

We'll use [jq](https://jqlang.github.io/jq/), a lightweight and flexible command-line JSON processor, to parse and manipulate JSON data efficiently. This tool will come in handy when dealing with the responses from the different platforms we'll interact with.

You can install jq by following the instructions in the [official documentation](https://jqlang.github.io/jq/download/), and verify its installation by running:

```shell
jq --version
```

Finally, you can run your first `just` command to check whether you have the required dependencies correctly installed in your environment:

```shell
just dependencies
```

Assuming everything works correctly, your environment should be ready to go.