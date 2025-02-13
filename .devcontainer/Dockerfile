FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

# RUN apt-get update \
#     && apt-get -y install --no-install-recommends curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pipx install rust-just

