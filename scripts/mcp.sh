#!/bin/bash
docker build -t mcp-mlschool .devcontainer \
    && docker run -i --rm \
    -v /Users/svpino/dev/ml.school/ml.school:/workspaces/ml.school \
    -w /workspaces/ml.school \
    mcp-mlschool \
    uv run mlschool.py