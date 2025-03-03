# Building Machine Learning Systems

"Building Machine Learning Systems" is designed to teach you how to train, evaluate, deploy, and monitor machine learning models in production. 

In this repository, you'll find the code to build a fully-fledged, end-to-end machine learning system that you can use as a starting point for your own projects.

This repository is part of the [Machine Learning School](https://www.ml.school) program.

## Running in a Development Container

The best way to clone and run the source code from this repository is using a Development Container.

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/svpino/ml.school)

Most Integrated Development Environments support Development Containers. If you have Visual Studio Code and Docker installed, you can click the badge above or [this link](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/svpino/ml.school) to automatically install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), clone the repository into a container volume, and spin up a container.

A Development Container is a Docker container configured as a fully functional development environment isolated from your operating system. You can use your IDE to edit, build, and run the project without spending time setting up your local environment or worrying about altering it.

You can find more information about Development Containers in the [Dev Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).

After opening the project on IDX, click on the "Machine Learning School" extension on the left activity bar. This extension will allow you to navigate the documentation, run the pipelines, and deploy the model.

**Note:** If you had the "Machine Learning School" extension installed before running the project on a Development Container, you'll need to uninstall it and rebuild the container. The extension must be installed on the container for it to work.

## Running on Google's Project IDX

An alternative way to run the project is using Google's Project IDX. Clicking the button below will create and configure a development environment you can access directly from your browser:

<a href="https://idx.google.com/new?template=https%3A%2F%2Fgithub.com%2Fsvpino%2Fml.school%2F">
  <img
    height="32"
    alt="Open in IDX"
    src="https://cdn.idx.dev/btn/open_dark_32.svg">
</a>

After opening the project on IDX, click on the "Machine Learning School" extension on the left activity bar. This extension will allow you to navigate the documentation, run the pipelines, and deploy the model.

*Note:* Project IDX is an experimental Google product and it might be unstable at times. If you are planning to take full advantage of this repository, and modify it for your own purposes, running in a Development Container is the best option.

## Running the project locally

If you prefer to run the project on your local environment, you can start by 
[forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the [repository](https://github.com/svpino/ml.school) and [cloning](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) it on your computer. 

You can run the code on any Unix-based operating system (e.g., Ubuntu or macOS). If you are using Windows, install the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about) (WSL).

Open the repository using Visual Studio Code and install the ["Machine Learning School"](https://marketplace.visualstudio.com/items?itemName=tideily.mlschool) extension. If you are using WSL, you need to install the extension on the WSL environment.

Once installed, this extension will allow you to navigate the documentation, run the pipelines, and deploy the model directly from Visual Studio Code.

## Contributing

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.