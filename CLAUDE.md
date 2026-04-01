# ml.school

Production-ready ML pipelines, agents, and MLOps practices using Metaflow, MLflow, TensorFlow/Keras, and Google ADK.

## Project Structure

```
src/
├── agents/          # AI agents (RAG, tic-tac-toe)
├── common/          # Shared utilities and pipeline components
├── inference/       # Model inference and serving
├── pipelines/       # Metaflow-based pipelines
└── scripts/         # Utility scripts and tools

tests/               # Test suites (mirrors src/ structure)
config/              # Pipeline configuration files
cloud-formation/     # AWS CloudFormation templates
notebooks/           # Jupyter notebooks
.guide/              # Educational documentation and tutorials
```

**Important**: The project uses `PYTHONPATH=src` for module imports.

## Prerequisites

- Python >=3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [just](https://github.com/casey/just) task runner

## Python Package Management With uv

Use `uv` exclusively for Python package management. Never use pip, pip-tools, poetry, or conda.

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`
- Run scripts: `uv run <script>.py`
- Run tools: `uv run pytest`, `uv run ruff`

## Development Commands

The `justfile` is the primary way to run project tasks. Run `just` to see all available commands.

### Common commands

```shell
just test              # Run unit tests
just train             # Run training pipeline
just mlflow            # Start MLflow server
just env               # Set up local .env file
just serve             # Serve latest model locally
just sample            # Send sample request to local model
just traffic           # Generate fake traffic for monitoring
just labels            # Generate fake labels
just monitor           # Run monitoring pipeline
```

AWS commands are grouped under the `aws` prefix (e.g., `just aws-setup`, `just sagemaker-deploy`).

## Environment Setup

1. Clone the repository
2. Install uv and just
3. Sync dependencies: `uv sync`
4. Set up environment variables: `just env` (creates `.env` from defaults)

The file `env.shared` contains the shared environment variable defaults. The `justfile` loads `.env` automatically via `set dotenv-load`.

### MLflow Tracking

- Start the server: `just mlflow`
- Access the UI at `http://localhost:5000`

## Linting and Formatting

Ruff is configured in `pyproject.toml` with these key settings:

- Line length: 88
- All rules enabled (`select = ["ALL"]`) with specific ignores
- Double quotes, space indentation
- Per-file ignores: `D103` in tests, `PLC0415` in pipelines, `F401` in `__init__.py`

Run: `uv run ruff check .` and `uv run ruff format .`

## Testing

Tests use pytest, configured in `pyproject.toml`:

- **Integration tests are excluded by default** — run them with `uv run pytest -m integration`
- `asyncio_mode = "auto"` — async tests work without manual decorators
- `--maxfail=2` — stops after 2 failures
- Fixtures live in `tests/pipelines/conftest.py` and `tests/inference/conftest.py`

Run tests: `just test` or `uv run pytest`

## Branch Strategy

- `main`: Production-ready code
- Feature branches: Use descriptive names (e.g., `feature/new-agent`, `fix/pipeline-bug`)

## Commit Guidelines

Write concise commit messages that focus on the "why" rather than the "what".

## Troubleshooting

**Import Errors**: Ensure you're using `uv run` for script execution and dependencies are synced with `uv sync`. The project relies on `PYTHONPATH=src`.

**Metaflow Issues**: Verify installation with `uv run python -c "import metaflow; print('OK')"`. Check AWS credentials if using remote execution.
