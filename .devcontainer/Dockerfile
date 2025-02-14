FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

RUN apt-get update \
    && apt-get -y install --no-install-recommends sqlite3

# Install just
RUN pipx install rust-just
