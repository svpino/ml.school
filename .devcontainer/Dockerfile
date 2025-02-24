FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye as base

RUN touch /etc/apt/sources.list.d/amd64.list
RUN dpkg --add-architecture amd64
RUN apt update

RUN apt-get update \
    && apt-get -y install --no-install-recommends binutils:amd64 libwebkit2gtk-4.0-37:amd64 sqlite3

# Install `just`
RUN pipx install rust-just

# Install uv and increase the timeout (in seconds) for HTTP requests
RUN pipx install uv
ENV UV_HTTP_TIMEOUT=300 
ENV UV_LINK_MODE=copy


RUN if [ "$(uname -m)" = "x86_64" ]; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws/ awscliv2.zip; \
    fi

RUN if [ "$(uname -m)" = "aarch64" ]; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws/ awscliv2.zip; \
    fi

