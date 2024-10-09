# Building Machine Learning Systems That Don't Suck

This repository contains the source code of the [Machine Learning
School](https://www.ml.school) program.

If you find any problems with the code or have any ideas on improving it, please open an
issue and share your recommendations.

## Table of Contents

-   [Preparing Your Environment](#preparing-your-environment)
-   [Running MLflow](#running-mlflow)
-   [Training The Model](#training-the-model)
-   [Visualizing Pipeline Results](#visualizing-pipeline-results)
-   [Deploying The Model](#deploying-the-model)
-   [Monitoring The Model](#monitoring-the-model)
-   [Production Pipelines in Amazon Web Services](#production-pipelines-in-amazon-web-services)
    -   [Running a remote MLflow server in EC2](#running-a-remote-mlflow-server-in-ec2)
    -   [Deploying the model to SageMaker](#deploying-the-model-to-sagemaker)
    -   [Monitoring the model in SageMaker](#monitoring-the-model-in-sagemaker)
    -   [Deploying to AWS Managed Services](#deploying-to-aws-managed-services)
    -   [Cleaning up AWS resources](#cleaning-up-aws-resources)
-   [Production Pipelines in Azure](#production-pipelines-in-azure)

## Preparing Your Environment

The program's code runs on any Unix-based operating system (e.g., Ubuntu or
macOS). If you are using Windows, install the [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/about) (WSL).

Start by forking the program's [GitHub
Repository](https://github.com/svpino/ml.school) and cloning it on your local
computer. This will allow you to modify your copy of the code and push the
changes to your repository.

The code in the repository was written using **Python 3.12**, so make sure you
have this [version](https://www.python.org/downloads/release/python-3126/)
installed in your environment. A more recent version of Python should also
work, but sticking to the same version will avoid any potential issues.

After cloning the repository, navigate to the root directory and create and
activate a virtual environment. We'll install all the required libraries inside
this virtual environment, preventing any conflicts with other projects you
might have on your computer:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Now, within the virtual environment, you can update `pip` and install the
libraries specified in the `requirements.txt` file:

```bash
pip3 install -U pip && pip3 install -r requirements.txt
```

At this point, you should have a working Python environment with all the
required dependencies. The final step is to create an `.env` file inside the
repository's root directory. We'll use this file to define the environment
variables we need to run the pipelines:

```bash
echo "KERAS_BACKEND=jax" >> .env
```

Finally, we'll use the [`jq`](https://jqlang.github.io/jq/) command-line JSON
processor to simplify some commands when working with different cloud
environments and [`docker`](https://docs.docker.com/engine/install/) to deploy
the model to the cloud. Make sure you have both tools installed in your system.

## Running MLflow

MLflow is a platform-agnostic machine learning lifecycle management tool that
will help you track experiments and share and deploy models.

To run an MLflow server locally, open a terminal window, activate the virtual
environment you created earlier, and run the following command:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Once running, you can navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in
your web browser to open MLflow's user interface.

By default, MLflow tracks experiments and stores data in files inside a local
`./mlruns` directory. You can change the location of the tracking directory or
use a SQLite database using the parameter `--backend-store-uri`. The following
example uses a SQLite database to store the tracking data:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite://mlflow.db
```

For more information on the MLflow server, run the following command:

```bash
mlflow server --help
```

After the server is running, modify the `.env` file inside the repository's
root directory to add the `MLFLOW_TRACKING_URI` environment variable pointing
to the tracking URI of the MLflow server. The following command will append the
variable to the file and export it in your current shell:

```bash
export $((echo "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" >> .env; cat .env) | xargs)
```

## Training The Model

The training pipeline trains, evaluates and registers a model in the [MLflow
Model Registry](https://mlflow.org/docs/latest/model-registry.html). We'll use
[Metaflow](https://metaflow.org), an open-source Python library, to orchestrate
the pipeline, run it, and track the data it generates.

In this section, we will run and orchestrate the training pipeline locally.
Later sections of this guide explain how to run the pipeline in a distributed
environment.

You can run the training pipeline using the following command from the
repository's root directory:

```bash
python3 pipelines/training.py --environment=pypi run
```

This pipeline will load and transform the `./data/penguins.csv` dataset, train
a model, use cross-validation to evaluate its performance and register the
model in the MLflow Model Registry. After the pipeline runs, you should see a
new version of the `penguins` model in the Model Registry.

The pipeline will register the model only if its accuracy is above a predefined
threshold. By default, the threshold is set to `0.7`, but you can change it by
specifying the `accuracy-threshold` parameter when running the pipeline:

```bash
python3 pipelines/training.py --environment=pypi run \
    --accuracy-threshold 0.9
```

The example above will only register the model if its accuracy is above 90%.

You can show the supported parameters for the Training flow by running the
following command:

```bash
python3 pipelines/training.py --environment=pypi run --help
```

## Visualizing Pipeline Results

We can observe the execution of each pipeline and visualize their results live
using [Metaflow Cards](https://docs.metaflow.org/metaflow/visualizing-results).
Metaflow will set up a local server for viewing these cards as the pipeline
runs.

For example, to open the built-in viewer for the Training pipeline, navigate to
your repository's root directory in a new terminal window and run this command:

```bash
python3 pipelines/training.py --environment=pypi card server
```

Open your browser and navigate to [localhost:8324](http://localhost:8324/).
Every time you run the Training pipeline, the viewer will automatically update
to show the cards related to the latest pipeline execution.

Check [Using Local Card
Viewer](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards#using-local-card-viewer)
for more information about the local card viewer.

## Deploying The Model
#TODO: Explain sqlite database to capture data.

To deploy your model locally, you can use the `mflow models serve` command
specifying the model version you want to deploy from the Model Registry. You
can find more information about local deployments in [Deploy MLflow Model as a
Local Inference
Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

The command below starts a local server listening in port `8080`. The server
hosts the latest version of the model in the Model Registry:

```bash
mlflow models serve \
    -m models:/penguins/$(
        curl -s -X GET "$MLFLOW_TRACKING_URI""/api/2.0/"\
"mlflow/registered-models/"\
"get-latest-versions" \
            -H "Content-Type: application/json" \
            -d '{"name": "penguins"}' | \
        jq -r '.model_versions[0].version'
    ) -h 0.0.0.0 -p 8080 --no-conda
```

After the server starts running, you can test the model by sending a request
with a sample input. The following command will output the prediction for the
given input:

```bash
curl -X POST http://0.0.0.0:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{
            "island": "Biscoe",
            "culmen_length_mm": 48.6,
            "culmen_depth_mm": 16.0,
            "flipper_length_mm": 230.0,
            "body_mass_g": 5800.0,
            "sex": "MALE"
        }]}'
```

## Monitoring The Model

The Monitoring pipeline monitors the performance of a hosted model. It runs a series of tests and generates several reports using the data captured by the model and a reference dataset.

To test the Monitoring pipeline, you can generate fake traffic to the hosted model. The model will capture the input data and the predictions it generates. You can then label that data to determine the model's performance.

Start by running the following pipeline to generate some traffic to the model:

```bash
python3 pipelines/traffic.py --environment=pypi run \
    --endpoint-uri=http://127.0.0.1:8080/invocations \
    --samples 200 \
    --drift True
```

The `--drift` parameter introduces a small drift in the data sent to the model. This is useful to test the Monitoring pipeline's ability to detect changes in the data. The `--samples` parameter specifies the number of samples to send to the model.

After generating some traffic, you can run the Labeling pipeline to automatically generate fake ground truth labels for the data captured by the model. The `--datastore-uri` parameter points to the SQLite database where the model captures the input data and predictions:

```bash
python3 pipelines/labeling.py --environment=pypi run \
    --datastore-uri sqlite:///penguins.db
```

In a real-world scenario, you would have a proper labeling process in place, but for testing purposes, the above pipeline does what we need.

At this point, you can set up Metaflow's built-in viewer for the Monitoring pipeline. Run the command below and navigate in your browserto [localhost:8324](http://localhost:8324/):

```bash
python3 pipelines/monitoring.py --environment=pypi card server
```

Finally, run the Monitoring pipeline using the command below:

```bash
python3 pipelines/monitoring.py --environment=pypi run \
    --datastore-uri sqlite:///penguins.db
```

You will see every report generated by the pipeline in the built-in viewer opened in your browser.

## Production Pipelines in Amazon Web Services

In this section, we'll use Amazon Web Services (AWS) to run a remote MLflow
server, run and orchestrate the pipelines in the cloud, and host the model.

Start by [creating a new AWS account](https://aws.amazon.com/free/) if you don't have one.

After creating the account, navigate to the "CloudFormation" service in your
AWS console, click on the "Create stack" button, and select "With new resources
(standard)". On the "Specify template" section, upload the
`cloud-formation/mlschool-cfn.yaml` template file and click the "Next" button.
Specify a name for the stack and user account and follow the prompts to create
the stack. After a few minutes, the stack status will change to
"CREATE_COMPLETE," and you can open the "Outputs" tab to access the output
values you'll need during the next steps.

Modify the `.env` file in the repository's root directory to add the
`AWS_USERNAME`, `AWS_ROLE`, and `AWS_REGION` environment variables. Before
running the command below, replace the values within square brackets using the
outputs from the CloudFormation stack:

```bash
export $( (tee -a .env << EOF
AWS_USERNAME=[AWS_USERNAME]
AWS_ROLE=[AWS_ROLE]
AWS_REGION=[AWS_REGION]
EOF
) && cat .env | xargs)
```

[Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment and configure it using the command below:

```bash
aws configure --profile $AWS_USERNAME
```

The configuration tool will ask for the "Access Key ID", "Secret Access Key",
and "Region." You can get the "Access Key ID" and "Region" from the
CloudFormation stack "Outputs" tab. To get the "Secret Access Key", navigate to
the "Secrets Manager" service in your AWS console and retrieve the secret value
under the `/credentials/mlschool` key.

Finally, configure the command line interface to use the role created by the
CloudFormation template. Run the following command to update your local AWS
configuration:

```bash
cat << EOF >> ~/.aws/config

[profile mlschool]
role_arn = $AWS_ROLE
source_profile = $AWS_USERNAME
region = $AWS_REGION
EOF
```

At this point, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command, or you can export the `AWS_PROFILE` environment variable for the current session to make it the default profile:

```bash
export AWS_PROFILE=mlschool
```

To ensure the permissions are correctly set, run the following command to return a list of S3 buckets in your account:

```bash
aws s3 ls
```

### Running a remote MLflow server in EC2

To configure a remote MLflow server, we'll use a CloudFormation template to set up a remote instance on AWS where we can run the server. This template will create a `t2.micro` [EC2](https://aws.amazon.com/ec2/) instance running Ubuntu. This tiny computer has one virtual CPU and 1 GiB of RAM. Amazon offers [750 hours of free usage](https://aws.amazon.com/free/) every month for this instance type, which should be enough for you to complete the program without incurring any charges. To create the stack, run the following command from the repository's root
directory:

```bash
aws cloudformation create-stack \
    --stack-name mlflow \
    --template-body file://cloud-formation/mlflow-cfn.yaml
```

You can open the "CloudFormation" service in your AWS console to check the status of the stack. It will take a few minutes for the status to change from "CREATE_IN_PROGRESS" to "CREATE_COMPLETE". Once it finishes, run the following command to download the private key associated with the EC2 instance and save it as `mlschool.pem` in your local directory:

```bash
aws ssm get-parameters \
    --names "/ec2/keypair/$(
        aws cloudformation describe-stacks \
            --stack-name mlflow \
            --query "Stacks[0].Outputs[?OutputKey=='KeyPair'].OutputValue" \
            --output text
    )" \
    --with-decryption | python3 -c '
import json
import sys
o = json.load(sys.stdin)
print(o["Parameters"][0]["Value"])
' > mlschool.pem
```

Change the permissions on the private key file to ensure the file is not
publicly accessible:

```bash
chmod 400 mlschool.pem
```

At this point, you can open the "EC2" service in your AWS console, and go to
the "Instances" page to find the new instance you'll be using to run the MLflow
server. Wait for the instance to finish initializing, and run the following
`ssh` command to connect to it:

```bash
ssh -i "mlschool.pem" ubuntu@$(aws cloudformation \
    describe-stacks --stack-name mlflow \
    --query "Stacks[0].Outputs[?OutputKey=='PublicDNS'].OutputValue" \
    --output text)
```

The EC2 instance comes prepared with everything you need to run the MLflow
server. From the terminal connected to the remote instance, run the following
command to start the server, binding it to the public IP address of the
instance:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Once the server starts running, open a browser in your computer and navigate to
the instance's public IP address on port 5000 to make sure MLflow is running
correctly. You can find the public IP address associated to the EC2 instance
with the following command:

```bash
echo $(aws cloudformation describe-stacks --stack-name mlflow \
    --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" \
    --output text)
```

Finally, modify the value of the `MLFLOW_TRACKING_URI` environment variable in
the `.env` file inside your repository's root directory and point it to the
remote instance. The following command will update the variable and export it
in your current shell:

```bash
awk -v s="MLFLOW_TRACKING_URI=http://$(aws cloudformation \
    describe-stacks --stack-name mlflow \
    --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
    --output text)":5000 '
BEGIN {found=0}
$0 ~ /^MLFLOW_TRACKING_URI=/ {
    print s
    found=1
    next
}
{print}
END {
    if (!found) print s
}' .env > .env.tmp && mv .env.tmp .env && export $(cat .env | xargs)
```

When you are done using the remote server, delete the CloudFormation stack to
remove the instance and avoid unnecessary charges. Check the [Cleaning up AWS
resources](#cleaning-up-aws-resources) section for more information.

### Deploying the model to SageMaker

You can use the Deployment pipeline to deploy the latest model from the Model Registry to different deployment targets. The pipeline will connect to the specified target platform, create a new endpoint to host the model, and run a few samples to test that everything works as expected.

To deploy the model to SageMaker, you'll need access to `ml.m4.xlarge` instances. By default, the quota for most new accounts is zero, so you might need to request a quota increase. You can do this in your AWS account under "Service Quotas" > "AWS Services" > "Amazon SageMaker". Find `ml.m4.xlarge for endpoint usage` and request a quota increase of 8 instances.

Start by creating an environment variable with the endpoint name you want to create. The following command will append the variable to the `.env` file and export it in your current shell:

```bash
export $((echo "ENDPOINT_NAME=penguins" >> .env; cat .env) | xargs)
```

Before deploying the model to SageMaker, you must build a Docker image and push it to the [Elastic Container Registry](https://aws.amazon.com/ecr/) (ECR). You can do this by running the following command:

```bash
mlflow sagemaker build-and-push-container
```

**macOS users**: Before running the above command, open the Docker Desktop application and under Advanced Settings, select the option "Allow the default Docker socket to be used" and restart Docker.

Once the image finishes uploading, you can proceed to run the deployment pipeline from the repository's root directory:

```bash
python3 pipelines/deployment.py --environment=pypi run \
    --target sagemaker \
    --endpoint $ENDPOINT_NAME \
    --region $AWS_REGION
```

If you want SageMaker to automatically capture the input data and predictions from the endpoint, you can set the `--data-capture-destination-uri` parameter to the S3 location where you want SageMaker to store the captured data:

```bash
python3 pipelines/deployment.py --environment=pypi run \
    --target sagemaker \
    --endpoint $ENDPOINT_NAME \
    --region $AWS_REGION \
    --data-capture-destination-uri s3://mlschool/penguins/datastore
```

The Deployment pipeline will create a new SageMaker endpoint, deploy the model, and run a few samples through it to test it. You can also test the endpoint from your terminal using the following `awscurl` command:

```bash
awscurl --service sagemaker --region "$AWS_REGION" \
    $(aws sts assume-role --role-arn "$AWS_ROLE" \
        --role-session-name mlschool-session \
        --profile "$AWS_USERNAME" --query "Credentials" \
        --output json | \
        jq -r '"--access_key \(.AccessKeyId)
        --secret_key \(.SecretAccessKey)
        --session_token \(.SessionToken)"'
    ) -X POST -H "Content-Type: application/json" \
    -d '{
      "inputs": [{
        "island": "Biscoe",
        "culmen_length_mm": 48.6,
        "culmen_depth_mm": 16.0,
        "flipper_length_mm": 230.0,
        "body_mass_g": 5800.0,
        "sex": "MALE"
      }]
    }' \
    https://runtime.sagemaker."$AWS_REGION".amazonaws.com/endpoints/"$ENDPOINT_NAME"/invocations
```

The above command will return a JSON response with the prediction result for the provided sample input.

As soon as you are done with the SageMaker endpoint, delete it to avoid unnecessary costs. Check the [Cleaning up AWS resources](#cleaning-up-aws-resources) section for more information.

### Monitoring the model in SageMaker

The Monitoring pipeline supports monitoring models hosted in SageMaker. For more information on how the pipeline works, check the [Monitoring The Model](#monitoring-the-model) section.

Start by running the following pipeline to generate some fake traffic to the model. Notice how the `--target` parameter is set to `sagemaker` and the `--endpoint-uri` parameter points to the endpoint where the model is hosted:

```bash
python3 pipelines/traffic.py --environment=pypi run \
    --target=sagemaker \
    --endpoint-uri=$ENDPOINT_NAME \
    --samples 200 \
    --drift True
```

You can now run the Labeling pipeline to automatically generate fake ground truth labels for the data captured by the endpoint. The `--datastore-uri` parameter should point to the endpoint's data capture location and the `--ground-truth-uri` parameter should point to the S3 location where the labels will be generated:

```bash
python3 pipelines/labeling.py --environment=pypi run \
    --datastore-uri s3://mlschool/penguins/datastore \
    --ground-truth-uri s3://mlschool/penguins/ground-truth
```

Set up Metaflow's built-in viewer for the Monitoring pipeline by running the command below and navigating in your browserto [localhost:8324](http://localhost:8324/):

```bash
python3 pipelines/monitoring.py --environment=pypi card server
```

Finally, run the Monitoring pipeline using the command below:

```bash
python3 pipelines/monitoring.py --environment=pypi run \
    --datastore-uri s3://mlschool/penguins/datastore \
    --ground-truth-uri s3://mlschool/penguins/ground-truth
```

You will see every report generated by the pipeline in the built-in viewer opened in your browser.

### Deploying to AWS Managed Services

In this section, we'll use [AWS Batch](https://aws.amazon.com/batch/) to run
the pipelines and [AWS Step Functions](https://aws.amazon.com/step-functions/)
to orchestrate them. Since these services are fully managed by AWS, they will
require little maintenance and will be reliable and highly available.

We can run Metaflow pipelines in both _local_ and _shared_ modes. While the
_local_ mode is ideal for developing and testing pipelines, the _shared_ mode
is designed to run them in a [production
environment](https://docs.metaflow.org/production/introduction).

In _shared_ mode, the Metaflow Development Environment and the Production
Scheduler rely on a separate compute cluster to provision compute resources on
the fly. A central Metadata Service will track all executions, and their
results will be stored in a common Datastore. Check the [Metaflow Service
Architecture](https://outerbounds.com/engineering/service-architecture/) for
more information.

We can run the pipelines in _shared_ mode using AWS Batch as the Compute
Cluster and AWS Step Functions as the Production Scheduler. Check [Using AWS
Batch](https://docs.metaflow.org/scaling/remote-tasks/aws-batch) for useful
tips and tricks for running Metaflow on AWS Batch.

To get started, create a new CloudFormation stack named `metaflow` by following
the [AWS Managed with
CloudFormation](https://outerbounds.com/engineering/deployment/aws-managed/cloudformation/)
instructions.

After the Cloud Formation stack is created, you can [configure the Metaflow
client](https://outerbounds.com/engineering/operations/configure-metaflow/)
using the information from the CloudFormation stack outputs. The command below
will configure Metaflow with a profile named `production` using the appropriate
configuration:

```bash
mkdir -p ~/.metaflowconfig && aws cloudformation describe-stacks \
    --stack-name metaflow \
    --query "Stacks[0].Outputs" \
    --output json | \
jq 'map({(.OutputKey): .OutputValue}) | add' | \
jq --arg METAFLOW_SERVICE_AUTH_KEY "$(
    aws apigateway get-api-key \
        --api-key $(
            aws cloudformation describe-stacks \
                --stack-name metaflow \
                --query "Stacks[0].Outputs[?OutputKey=='ApiKeyId'].OutputValue" \
                --output text
        ) \
        --include-value \
        --output json | jq -r '.value'
)" '{
    "METAFLOW_BATCH_JOB_QUEUE": .BatchJobQueueArn,
    "METAFLOW_DATASTORE_SYSROOT_S3": .MetaflowDataStoreS3Url,
    "METAFLOW_DATATOOLS_S3ROOT": .MetaflowDataToolsS3Url,
    "METAFLOW_DEFAULT_DATASTORE": "s3",
    "METAFLOW_DEFAULT_METADATA": "service",
    "METAFLOW_ECS_S3_ACCESS_IAM_ROLE": .ECSJobRoleForBatchJobs,
    "METAFLOW_EVENTS_SFN_ACCESS_IAM_ROLE": .EventBridgeRoleArn,
    "METAFLOW_SERVICE_AUTH_KEY": $METAFLOW_SERVICE_AUTH_KEY,
    "METAFLOW_SERVICE_INTERNAL_URL": .InternalServiceUrl,
    "METAFLOW_SERVICE_URL": .ServiceUrl,
    "METAFLOW_SFN_DYNAMO_DB_TABLE": .DDBTableName,
    "METAFLOW_SFN_IAM_ROLE": .StepFunctionsRoleArn
}' > ~/.metaflowconfig/config_production.json
```

To keep using Metaflow in _local_ mode, you must run the following command to
create a new profile with an empty configuration. You can check [Use Multiple
Metaflow Configuration
Files](https://docs.outerbounds.com/use-multiple-metaflow-configs/)
for more information:

```bash
echo '{}' > ~/.metaflowconfig/config_local.json
```

You can now enable the profile you want to use when running the pipelines by
exporting the `METAFLOW_PROFILE` variable in your local environment. For
example, to run your pipelines in _shared_ mode, you can set the environment
variable to `production`:

```bash
export METAFLOW_PROFILE=production
```

You can also prepend the profile name to the command running the pipeline. For
example, to run the Training pipeline in _local_ mode, you can use the
following:

```bash
METAFLOW_PROFILE=local python3 pipelines/training.py \
    --environment=pypi run
```

Remember to delete the `metaflow` CloudFormation stack as soon as you are done
using it to avoid unnecessary charges. Check the [Cleaning up AWS
resources](#cleaning-up-aws-resources) section for more information.

#### Running the Training pipeline remotely

You can now run the Training pipeline remotely by using the `--with batch` and
`--with retry` parameters. These will mark every step of the flow with the
`batch` and `retry` decorators, They will instruct Metaflow to run every step
in AWS Batch and retry them if they fail:

```bash
python3 pipelines/training.py --environment=pypi run \
    --with batch --with retry
```

For the command above to work, remember to set the `METAFLOW_PROFILE`
environment variable to `production` in your shell.

At this point, the pipeline will run in a remote compute cluster but it will
still use the local environment to orchestrate the workflow. You can [schedule
the pipeline using AWS Step
Functions](https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-aws-step-functions)
using the command below:

```bash
python3 pipelines/training.py --environment=pypi step-functions create
```

The above command will take a snapshot of the pipeline code and deploy it to
AWS Step Functions. After you run the command, list the existing state machines
in your account and you'll see a new state machine associated with the Training
pipeline:

```bash
aws stepfunctions list-state-machines
```

To trigger the state machine corresponding to the Training pipeline, use the
`step-functions trigger` parameter:

```bash
python3 pipelines/training.py --environment=pypi step-functions trigger
```

The above command will create a new execution of the state machine and run the
Training pipeline in the remote compute cluster. You can check the status of
the execution under the Step Functions service in your AWS console or by
running the following command:

```bash
aws stepfunctions describe-execution \
    --execution-arn "$(
        aws stepfunctions list-executions \
            --state-machine-arn "$(
                aws stepfunctions list-state-machines \
                    --query "
                        stateMachines[?ends_with(name, '.Training')].stateMachineArn
                        | [0]
                    " \
                    --output text
            )" \
            --max-results 1 --no-paginate \
            --query "executions[0].executionArn" \
            --output text
    )"
```

#### Running the Deployment pipeline remotely

To run the Deployment pipeline in the remote compute cluster, you need to
modify the permissions associated with one of the roles created by the
`metaflow` CloudFormation stack. The new permissions will allow the role to
access the Elastic Container Registry (ECR) and deploy the model to SageMaker:

```bash
aws iam put-role-policy \
    --role-name "$(
        aws iam list-roles \
            --query "Roles[?contains(RoleName, '-BatchS3TaskRole-')].RoleName" \
            --output text
    )" \
    --policy-name mlschool \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "mlschool",
                "Effect": "Allow",
                "Action": [
                    "sts:AssumeRole",
                    "ecr:DescribeRepositories"
                ],
                "Resource": "*"
            }
        ]
    }'
```

At this point, you can run the Deployment pipeline in the remote compute
cluster using the following command:

```bash
python3 pipelines/deployment.py --environment=pypi run \
    --with batch \
    --target sagemaker \
    --region $AWS_REGION \
    --assume-role $AWS_ROLE
```

Notice you need to specify the role using the `--assume-role` parameter when
running the pipeline in the remote cluster. This role is authorized to create
the resources to host the model in SageMaker.

To deploy the Deployment pipeline to AWS Step Functions, you can use the
`step-functions create` parameter:

```bash
python3 pipelines/deployment.py --environment=pypi step-functions create
```

To trigger the state machine corresponding to the Deployment pipeline, use the
`step-functions trigger` parameter:

```bash
python3 pipelines/deployment.py --environment=pypi step-functions trigger \
    --target sagemaker \
    --region $AWS_REGION \
    --assume-role $AWS_ROLE
```

Finally, you can check the status of the execution by running the command
below:

```bash
aws stepfunctions describe-execution \
    --execution-arn "$(
        aws stepfunctions list-executions \
            --state-machine-arn "$(
                aws stepfunctions list-state-machines \
                    --query "
                        stateMachines[?ends_with(name, '.Deployment')].stateMachineArn
                        | [0]
                    " \
                    --output text
            )" \
            --max-results 1 --no-paginate \
            --query "executions[0].executionArn" \
            --output text
    )"
```

### Cleaning up AWS resources

When you finish using your AWS account, clean up everything to prevent
unnecessary charges.

The command below removes two of the CloudFormation stacks we deployed. These
stacks are responsible for the resources with associated costs:

```bash
echo "metaflow mlflow" \
  | xargs -n 1 -I {} aws cloudformation delete-stack --stack-name {}
```

If you aren't planning to return to the program, you can also remove the
CloudFormation stack configuring your account and permissions. Keep in mind
that the resources created by this stack do not cost money, so you can keep
them around indefinitely if you want:

```bash
aws cloudformation delete-stack --stack-name mlschool
```

Finally, you can run the following command to delete the endpoint from
SageMaker:

```bash
aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
```

## Production Pipelines in Azure

1. Create a [free Azure
   account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount)
   if you don't have one.

2. Install the Azure [Command Line Interface
   (CLI)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and [configure
   it on your
   environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public).
   After finishing these steps, you should be able to run the following command to display
   your Azure subscription and configuration:

```bash
az account show && az configure -l
```

3. In the Azure Portal, find the _Resource providers_ tab under your subscription.
   Register the `Microsoft.Cdn` and the `Microsoft.PolicyInsights` providers.

4. To deploy the model to an endpoint, we need to request a quota increase for the
   virtual machine we'll be using. In the Azure Portal, open the _Quotas_ tab and filter
   the list by the _Machine learning_ provider, your subscription, and your region. Request
   a quota increase for the `Standard DSv2 Family Cluster Dedicated vCPUs`. Set the new
   quota limit to 16.

5. If it doesn't exist, create an `.env` file inside the repository's root directory
   with the environment variables below. Make sure to replace `[AZURE_SUBSCRIPTION_ID]`
   with the appropriate values:

```bash

AZURE_SUBSCRIPTION_ID=[AZURE_SUBSCRIPTION_ID]
AZURE_RESOURCE_GROUP=mlschool
AZURE_WORKSPACE=main

```

6. Export the environment variables from the `.env` file in your current shell:

```bash
export $(cat .env | xargs)
```

#### Running the deployment pipeline

After you finish setting up your Azure account, you can run the deployment pipeline from
the repository's root directory using the following command:

```bash
python3 pipelines/deployment.py --environment=pypi \
    run --target azure \
    --endpoint $ENDPOINT_NAME
```

For more information on the deployment pipeline and the parameters you can use to
customize it, you can run the following command:

```bash
python3 pipelines/deployment.py --environment=pypi run --help
```

After you are done with the Azure endpoint, make sure you delete it to avoid unnecessary
costs:

```bash
az ml online-endpoint delete --name $ENDPOINT_NAME \
    --resource-group $AZURE_RESOURCE_GROUP \
    --workspace-name $AZURE_WORKSPACE \
    --yes
```

You can also delete the entire resource group if you aren't planning to use it anymore.
This will delete all the resources you created to host the model:

```bash
az group delete --name $AZURE_RESOURCE_GROUP
```
