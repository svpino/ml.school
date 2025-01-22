set dotenv-load
set positional-arguments

KERAS_BACKEND := env("KERAS_BACKEND", "jax")
MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

default:
    @just --list

# Run project unit tests
test:
    uv run -- pytest

# Display version of required dependencies
[group('setup')]
@dependencies:
    uv_version=$(uv --version) && \
        just_version=$(just --version) && \
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//') && \
        jq_version=$(jq --version | awk -F'-' '{print $2}') && \
    echo "uv: $uv_version" && \
    echo "just: $just_version" && \
    echo "docker: $docker_version" && \
    echo "jq: $jq_version"

# Run MLflow server
[group('setup')]
@mlflow:
    uv run -- mlflow server --host 127.0.0.1 --port 5000

# Set up required environment variables
[group('setup')]
@env:
    echo "KERAS_BACKEND={{KERAS_BACKEND}}\nMLFLOW_TRACKING_URI={{MLFLOW_TRACKING_URI}}" > .env
    cat .env


# Run training pipeline
[group('training')]
@train:
    uv run -- python pipelines/training.py \
        --environment conda run

# Run training pipeline card server 
[group('training')]
@train-viewer:
    uv run -- python pipelines/training.py \
        --environment conda card server

# Serve latest registered model locally
[group('serving')]
@serve:
    uv run -- mlflow models serve \
        -m models:/penguins/$(curl -s -X GET "{{MLFLOW_TRACKING_URI}}/api/2.0/mlflow/registered-models/get-latest-versions" \
        -H "Content-Type: application/json" -d '{"name": "penguins"}' \
        | jq -r '.model_versions[0].version') -h 0.0.0.0 -p 8080 --no-conda

# Invoke local running model with sample request
[group('serving')]
@invoke:
    uv run -- curl curl -X POST http://0.0.0.0:8080/invocations \
        -H "Content-Type: application/json" \
        -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE" }]}'

# Display number of records in SQLite database
[group('serving')]
@sqlite:
    uv run -- sqlite3 penguins.db "SELECT COUNT(*) FROM data;"

# Generate fake traffic to local running model
[group('monitoring')]
@traffic:
    uv run -- python pipelines/traffic.py \
        --environment conda run \
        --samples 200

# Generate fake labels in SQLite database
[group('monitoring')]
@labels:
    uv run -- python pipelines/labels.py \
        --environment conda run

# Run the monitoring pipeline
[group('monitoring')]
@monitor:
    uv run -- python pipelines/monitoring.py \
        --config backend config/local.json \
        --environment conda run

# Run monitoring pipeline card server 
[group('monitoring')]
@monitor-viewer:
    uv run -- python pipelines/monitoring.py \
        --environment conda card server \
        --port 8334

# Deploy model to Sagemaker
[group('sagemaker')]
@sagemaker-deploy:
    uv run -- python pipelines/deployment.py \
        --config backend config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Invoke Sagemaker endpoint with sample request
[group('sagemaker')]
@sagemaker-invoke:
    awscurl --service sagemaker --region "$AWS_REGION" \
        $(aws sts assume-role --role-arn "$AWS_ROLE" \
            --role-session-name mlschool-session \
            --profile "$AWS_USERNAME" --query "Credentials" \
            --output json | \
            jq -r '"--access_key \(.AccessKeyId) --secret_key \(.SecretAccessKey) --session_token \(.SessionToken)"' \
        ) -X POST -H "Content-Type: application/json" \
        -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE" }] }' \
        https://runtime.sagemaker."$AWS_REGION".amazonaws.com/endpoints/"$ENDPOINT_NAME"/invocations


# Delete Sagemaker endpoint
[group('sagemaker')]
@sagemaker-delete:
    aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT_NAME"

# Generate fake traffic to Sagemaker endpoint
[group('sagemaker')]
@sagemaker-traffic:
    uv run -- python pipelines/traffic.py \
        --config backend config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker \
        --samples 200

# Generate fake labels in SQLite database
[group('sagemaker')]
@sagemaker-labels:
    uv run -- python pipelines/labels.py \
        --config backend config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

[group('sagemaker')]
@sagemaker-monitor-viewer:
    uv run -- python pipelines/monitoring.py \
        --environment conda card server \

[group('sagemaker')]
@sagemaker-monitor:
    uv run -- python pipelines/monitoring.py \
        --config backend config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker