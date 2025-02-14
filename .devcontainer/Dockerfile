FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

RUN apt-get update \
    && apt-get -y install --no-install-recommends sqlite3

# Install `just`
RUN pipx install rust-just

# Install uv and increase the timeout (in seconds) for HTTP requests
RUN pipx install uv
ENV UV_HTTP_TIMEOUT=300 

# Install the AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf aws/ awscliv2.zip

