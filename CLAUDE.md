# Building AI/ML Systems That Don't Suck - ml.school

## Overview

"Building AI/ML Systems That Don't Suck" is a project designed to teach you how to train, evaluate, deploy, and monitor AI and Machine Learning models in production. The project demonstrates production-ready pipelines, agents, and MLOps best practices using modern tools and frameworks.

**Key Components:**
- **Pipelines**: Training, monitoring, and deployment pipelines built with Metaflow
- **AI Agents**: Interactive agents including RAG systems using Google ADK
- **Infrastructure**: AWS integration, MLflow tracking, and monitoring systems
- **Documentation**: Comprehensive guides for Metaflow, pipelines, and project-related topics

## Project Structure

```
.guide/              # Educational documentation and tutorials
├── introduction/                # Introduction to the program
├── introduction-to-metaflow/    # Metaflow learning materials
├── training-pipeline/           # Training pipeline documentation
├── inference-pipeline/          # Inference pipeline documentation
├── monitoring-pipeline/         # Monitoring pipeline documentation
├── serving-model/               # Model serving documentation
└── amazon-web-services/         # AWS deployment documentation

src/
├── agents/          # AI agents (RAG, tic-tac-toe, etc.)
├── common/          # Shared utilities and pipeline components
├── inference/       # Model inference and serving
├── pipelines/       # Metaflow-based pipelines
└── scripts/         # Utility scripts and tools

tests/               # Test suites
```

**Important**: The project uses `PYTHONPATH=src` for module imports.

## Python Package Management With uv

Use `uv` exclusively for Python package management in this project.

### Requirements
- Python >=3.12
- uv package manager

### Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

### Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Development Commands

### Pipelines (Metaflow)
```shell
# Run a training pipeline
uv run src/pipelines/training.py run

# Run with parameters
uv run src/pipelines/training.py run --max_epochs 100

# View pipeline results
uv run src/pipelines/training.py card view

# Show pipeline structure
uv run src/pipelines/training.py show
```

### Agents and Scripts
```shell
# Run ML School CLI
uv run src/scripts/mlschool.py --help

# Run tic-tac-toe tournament
uv run src/agents/tic_tac_toe/tic_tac_toe/agent.py

# Test RAG system
uv run src/agents/rag/agent.py
```

## Environment Setup

### Local Development
1. Clone the repository
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Sync dependencies: `uv sync`
4. Set up environment variables as needed (see `.env.example` if available)

### MLflow Tracking
- MLflow is configured for experiment tracking
- Access the UI at `http://localhost:5000` when running locally
- See training pipeline documentation for usage examples


### Branch Strategy
- `main`: Production-ready code
- Feature branches: Use descriptive names (e.g., `feature/new-agent`, `fix/pipeline-bug`)
- Current development branch: Check `git branch` for active work

### Commit Guidelines
- Before commiting changes, **ALWAYS** use the @.claude/skills/generating-commit-messages skill to generate commit messages

## Troubleshooting

### Common Issues

**Import Errors:**
- Ensure `PYTHONPATH=src` is set
- Check that you're using `uv run` for script execution
- Verify dependencies are synced with `uv sync`

**Test Failures:**
- Run `uv sync` to ensure all dependencies are installed
- Check that test data files exist in expected locations
- Verify environment variables are set correctly

**Metaflow Issues:**
- Ensure Metaflow is properly configured for your environment
- Check AWS credentials if using remote execution
- Use `uv run python -c "import metaflow; print('OK')"` to verify installation

### Getting Help
- Check the `.guide/` directory for comprehensive documentation
- Review test files for usage examples
- Consult official documentation for Metaflow, MLflow, and other frameworks
- Use `uv run <script> --help` for command-line tool documentation

## Key Dependencies

- **Metaflow**: Workflow orchestration and pipeline management
- **MLflow**: Experiment tracking and model registry
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Traditional Machine Learning algorithms
- **Pandas/Numpy**: Data manipulation and analysis
- **AWS SDK (boto3)**: Cloud integration
- **Ruff**: Fast Python linter and formatter
- **Pytest**: Testing framework