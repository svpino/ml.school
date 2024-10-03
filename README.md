# Building Machine Learning Systems That Don't Suck

This repository contains the source code of the [Machine Learning
School](https://www.ml.school) program.

If you find any problems with the code or have any ideas on improving it, please open an
issue and share your recommendations.

## Preparing Your Local Environment

TBD

The code in the program runs on any Unix-based operating system (e.g. Ubuntu or macOS).
If you are using Windows, install the [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/about) (WSL).

Start by forking the program's [GitHub
Repository](https://github.com/svpino/ml.school) and cloning it on your local
computer. This will allow you to modify your copy of the code and push the
changes to your personal repository.

The code in the repository was written using **Python 3.12**, so make sure you
have this [Python
version](https://www.python.org/downloads/release/python-3126/) installed in
your environment. A more recent version of Python should work as well, but
sticking to the same version will avoid any potential issues.

After cloning the repository, navigate to the main directory and create and
activate a virtual environment. We'll install all the required libraries inside
this virtual environment, preventing any conflicts with other projects you
might have on your computer:

```bash
python3 -m venv .venv 
source .venv/bin/activate
```

Now, within the virtual environment, you can update `pip` and install the libraries
specified in the `requirements.txt` file:

```bash
pip3 install -U pip && pip3 install -r requirements.txt
```

At this point, you should have a working Python environment with all the required
dependencies. The final step is to create an `.env` file inside the repository's root
directory. We'll use this file to define a few environment variables we'll need to run
the pipelines:

```bash
cat << EOF >> .env
KERAS_BACKEND=jax
ENDPOINT_NAME=penguins
EOF
```

The `KERAS_BACKEND` environment variable specifies the framework
[Keras](https://keras.io/) will use when training a model. The `ENDPOINT_NAME`
variable specifies the name of the endpoint we'll use to host the model in the
cloud.

Finally, we'll use the [`jq`](https://jqlang.github.io/jq/) command-line JSON
processor to simplify some of the commands when working with different cloud
environments, and [`docker`](https://docs.docker.com/engine/install/) to deploy
the model to the cloud. If you don't have them already, install both tools in
your environment.

## Setting Up Amazon Web Services

We'll use Amazon Web Services (AWS) at different points in the program to run the
pipelines in the cloud, host the model, and run a remote MLflow server.

Start by [creating a new AWS account](https://aws.amazon.com/free/) if you don't have one.

After creating the account, navigate to the "CloudFormation" service in your AWS
console, click on the "Create stack" button and select "With new resources (standard)".
On the "Specify template" section, upload the `cloud-formation/mlschool-cfn.yaml`
template file and click on the "Next" button. Specify a name for the stack and a name
for a user account and follow the prompts to create the stack. After a few minutes, the
stack status will change to "CREATE_COMPLETE" and you'll be able to open the "Outputs"
tab to access the output values you'll need during the next steps.

Modify the  `.env` file in the repository's root directory to add the
`AWS_USERNAME`, `AWS_ROLE`, and `AWS_REGION` environment variables. Before
running the command below, replace the values within square brackets using the
outputs from the CloudFormation stack:

```bash
cat << EOF >> .env
AWS_USERNAME=[AWS_USERNAME]
AWS_ROLE=[AWS_ROLE]
AWS_REGION=[AWS_REGION]
EOF
```

You can now export the environment variables from the `.env` file in your current shell:

```bash
export $(cat .env | xargs)
```

[Install the AWS
CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
on your environment and configure it using the command below:

```bash
aws configure --profile $AWS_USERNAME
```

The configuration tool will ask for the "Access Key ID", "Secret Access Key", and "Region."
You can get the "Access Key ID" and "Region" from the CloudFormation stack
"Outputs" tab. To get the "Secret Access Key", navigate to the "Secrets Manager" service
in your AWS console and retrieve the secret value under the `/credentials/mlschool`
key.

Finally, configure the command line interface to use the role created by the
CloudFormation template. Run the following command to update the local AWS
configuration:

```bash
cat << EOF >> ~/.aws/config

[profile mlschool]
role_arn = $AWS_ROLE
source_profile = $AWS_USERNAME
region = $AWS_REGION
EOF
```

At this point, you should be able to take advantage of the role's permissions at the
command line by using the `--profile` option on every AWS command. For example, the
following command should return a list of S3 buckets in your account:

```bash
aws s3 ls --profile mlschool
```

To take advantage of the role's permissions on every command without having to specify
the `--profile` option, export the `AWS_PROFILE` environment variable for the current
session:

```bash
export AWS_PROFILE=mlschool
```

You can verify that the `mlschool` profile is being used by running the following
command and looking at the `Arn` associated to your identity. This value should point to
the role we created using the CloudFormation template:

```bash
aws sts get-caller-identity
```

## Setting Up MLflow

MLflow is a platform-agnostic machine learning lifecycle management tool that will help
us track experiments and share and deploy models. For this program, you can run the
MLflow server locally to keep everything in your local computer. For a more scalable
solution, you should run MLflow from a remote server.

* [Running MLflow locally](#running-mlflow-locally)
* [Running MLflow in a remote server](#running-mlflow-in-a-remote-server)

### Running MLflow locally

Open a terminal window, activate the virtual environment you created previously, and
install the `mlflow` library using the command below:

```bash
pip3 install mlflow
```

Once you install the `mlflow` library, you can run the server with the following command:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Once running, you can open [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your web
browser to see the user interface.

By default, MLflow tracks experiments and stores data in files inside a local `./mlruns`
directory. You can change the location of the tracking directory or use a SQLite
database to store the tracking data using the parameter `--backend-store-uri`:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite://mlflow.db
```

For more information on the MLflow server, run the following command:

```bash
mlflow server --help
```

After the server is running, modify the `.env` file inside the repository's root
directory and add the `MLFLOW_TRACKING_URI` environment variable pointing to the
tracking URI of the MLflow server. Make sure you also export this variable in your
current shell:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### Running MLflow in a remote server

To configure a remote MLflow server, we'll use a CloudFormation template to set up a
remote instance on AWS where we'll run the server. This template will create a
`t2.micro` EC2 instance running Ubuntu. This is a very small computer, with 1 virtual
CPU and 1 GiB of RAM, but Amazon offers [750 hours of free
usage](https://aws.amazon.com/free/) every month for this instance type, which should be
enough for you to complete the program without incurring any charges.

To create the stack, run the following command from the repository's root directory:

```bash
aws cloudformation create-stack \
    --stack-name mlschool-mlflow \
    --template-body file://cloud-formation/mlflow-cfn.yaml
```

You can open the "CloudFormation" service in your AWS console to check the status of the
stack. It will take a few minutes for the status to change from "CREATE_IN_PROGRESS" to
"CREATE_COMPLETE". Once it finishes, run the following command to grab the output values
you'll need in the following steps:

```bash
read keypair publicdns <<< $(aws cloudformation \
    describe-stacks --stack-name mlschool-mlflow \
    --query "join(' ', Stacks[0].Outputs[?OutputKey=='KeyPair' || OutputKey=='PublicDNS'].OutputValue)" \
    --output text)
```

You can now download the private key associated with the EC2 instance and save it as
`mlschool.pem` in your local directory:

```bash
aws ssm get-parameters --names "/ec2/keypair/$keypair" \
    --with-decryption | python3 -c 'import json;import sys;o=json.load(sys.stdin);print(o["Parameters"][0]["Value"]);' \
    > mlschool.pem
```

Change the permissions on the private key file to ensure the file is not publicly accessible:

```bash
chmod 400 mlschool.pem
```

At this point, you can open the "EC2" service, and go to the "Instances" page to find
the new instance you'll be using to run the MLflow server. Wait for the instance to
finish initializing, and run the following `ssh` command to connect to it:

```bash
ssh -i "mlschool.pem" ubuntu@$publicdns
```

The EC2 instance comes prepared with everything you need to run the MLflow server, so
you can run the following command to start and bind the server to the public IP address
of the instance:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Once the server starts running, open a browser and navigate to the instance's public IP
address on port 5000. This will open the MLflow user interface. You can find the public
IP address associated to the EC2 instance with the following command:

```bash
echo $(aws cloudformation describe-stacks --stack-name mlschool-mlflow \
    --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" \
    --output text)
```

Finally, modify the `.env` file inside the repository's root directory to add the
`MLFLOW_TRACKING_URI` environment variable. This variable should point to the URL of the
MLflow server, so replace `[PUBLIC_IP]` with the public IP address of the EC2 instance.
Make sure you also export this variable in your current shell:

```bash
MLFLOW_TRACKING_URI=http://[PUBLIC_IP]:5000
```

When you are done using the remote MLflow server, delete the CloudFormation stack to
avoid unnecessary charges:

```bash
aws cloudformation delete-stack --stack-name mlschool-mlflow
```

## Training The Model

The training pipeline trains, evaluates, and registers a model in the [MLflow Model
Registry](https://mlflow.org/docs/latest/model-registry.html). We'll use
[Metaflow](https://metaflow.org), an open-source Python library, to orchestrate the
pipeline, run it, and track the data it generates.

In this section, we will run the training pipeline locally. For information on how to
run the pipeline in a distributed environment, check the [Running Pipelines in
Production](#running-pipelines-in-production) section.

From the repository's root directory, run the training pipeline locally using the
following command:

```bash
python3 pipelines/training.py --environment=pypi run
```

This pipeline will load and transform the `./data/penguins.csv` dataset, train a model, use
cross-validation to evaluate its performance, and register the model in the MLflow Model
Registry. After the pipeline finishes running, you should see the new version of the
`penguins` model in the Model Registry.

The pipeline will register the model only if its accuracy is above a predefined threshold.
By default, the threshold is set to `0.7`, but you can change it by specifying the
`accuracy-threshold` parameter when running the flow:

```bash
python3 pipelines/training.py --environment=pypi run \
    --accuracy-threshold 0.9
```

The example above will only register the model if its accuracy is above 90%.

You can show the supported parameters for the Training flow by running the following command:

```bash
python3 pipelines/training.py --environment=pypi run --help
```

## Deploying The Model

The deployment pipeline deploys the latest model from the Model Registry to a
number of deployment targets.

Except when deploying the model locally, you can run the deployment pipeline
specifying the target platform using the `--target` parameter. The flow will
connect to the target platform, create a new endpoint to host the model, and
run inference using a few samples to test that everything works as expected.

For information on how to deploy the model to each supported deployment target, follow
the respective links below:

* [Deploying the model as a local inference server](#deploying-the-model-as-a-local-inference-server)
* [Deploying the model to SageMaker](#deploying-the-model-to-sagemaker)
* [Deploying the model to Azure Machine Learning](#deploying-the-model-to-azure-machine-learning)

### Deploying the model as a local inference server

To deploy your model locally, you can use the `mflow models serve` command specifying
the model version you want to deploy from the Model Registry. You can find more
information about local deployments in [Deploy MLflow Model as a Local Inference
Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

The command below starts a local server listening on the specified port and network
interface and uses the active Python environment to execute the model. Make sure you
replace `[MODEL_VERSION]` with the version of the model you want to deploy:

```bash
mlflow models serve -m models:/penguins/[MODEL_VERSION] \
    -h 0.0.0.0 -p 8080 \
    --no-conda
```

After the server starts running, you can test the model by sending a request with a
sample input. The following command should return a prediction for the provided input:

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

### Deploying the model to SageMaker

We can use the Deployment pipeline to deploy the latest version of the model to SageMaker.

To create an endpoint in SageMaker, you'll need access to `ml.m4.xlarge` instances. By
default, the quota for most new accounts is zero, so you might need to request a quota
increase. You can do this in your AWS account under "Service Quotas" > "AWS Services" >
"Amazon SageMaker". Find `ml.m4.xlarge for endpoint usage` and request a quota increase
of 8 instances.

Before we can deploy the model to SageMaker, we need to build a Docker image and push it
to the Elastic Container Registry (ECR) in AWS. You can do this by running the following
command:

```bash
mlflow sagemaker build-and-push-container
```

**macOS users**: Before running the above command, open the Docker Desktop
application and under Advanced Settings, select the option "Allow the default
Docker socket to be used" and restart Docker.

Once the image finishes uploading, you can proceed to run the deployment
pipeline from the repository's root directory:

```bash
python3 pipelines/deployment.py --environment=pypi run \
    --target sagemaker \
    --endpoint $ENDPOINT_NAME \
    --region $AWS_REGION
  ```

The pipeline will create a new SageMaker endpoint, deploy the model, and run a few
samples to test the endpoint.

After the pipeline finishes running, you can test the endpoint from your
terminal, using the `awscurl` command:

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

The above command will return a JSON response with the prediction result for
the provided input.

As soon as you are done with the SageMaker endpoint, make sure you delete it to avoid
unnecessary costs:

```bash
aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
```

### Deploying the model to Azure Machine Learning

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

3. In the Azure Portal, find the *Resource providers* tab under your subscription.
   Register the `Microsoft.Cdn` and the `Microsoft.PolicyInsights` providers.

4. To deploy the model to an endpoint, we need to request a quota increase for the
   virtual machine we'll be using. In the Azure Portal, open the *Quotas* tab and filter
the list by the *Machine learning* provider, your subscription, and your region. Request
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

## Monitoring The Model

TBD

## Running Pipelines in Production

[Production-grade workflows](https://docs.metaflow.org/production/introduction) should
be fully automated, reliable, and highly available. We can run Metaflow pipelines in
*local* and *shared* mode. While the *local* mode is ideal for developing and testing
pipelines, the *shared* mode is designed to run the pipelines in a production
environment.

In *shared* mode, the Metaflow Development Environment and the Production Scheduler rely
on a separate compute cluster to provision compute resources on the fly. A central
Metadata Service will track all executions and their results will be persisted in a
common Datastore. Check [Service
Architecture](https://outerbounds.com/engineering/service-architecture/) for more
information on the Metaflow architecture.

We have several options to run the Metaflow pipelines in production:

* [Deploying with Outerbounds](#deploying-with-outerbounds)
* [Deploying to AWS Managed Services](#deploying-to-aws-managed-services)

### Deploying with Outerbounds

TBD

### Deploying to AWS Managed Services

We can run the pipelines in *shared* mode using AWS Batch as the Compute Cluster and AWS
Step Functions as the Production Scheduler. Check [Using AWS
Batch](https://docs.metaflow.org/scaling/remote-tasks/aws-batch) for some useful tips
and tricks related to running Metaflow on AWS Batch.

To get started, create a new CloudFormation stack named `metaflow` by following the [AWS
Managed with
CloudFormation](https://outerbounds.com/engineering/deployment/aws-managed/cloudformation/)
instructions.

When the CloudFormation stack is created, you can access the outputs of the stack using
the following command:

```bash
aws cloudformation describe-stacks \
    --stack-name metaflow --query "Stacks[0].Outputs"
```

Remember to delete the CloudFormation stack as soon as you are done using it to avoid
unnecessary charges:

```bash
aws cloudformation delete-stack --stack-name metaflow
```

#### Configuring the Metaflow client

After the CloudFormation stack is created, fetch the API Gateway Key ID for the Metadata
Service using the command below. Replace `[API_KEY_ID]` with the stack "ApiKeyId"
output. You'll need this value to configure the `METAFLOW_SERVICE_AUTH_KEY` variable in
the next step:

```bash
aws apigateway get-api-key --api-key [API_KEY_ID] \
    --include-value | grep value
```

You can now [configure the Metaflow
client](https://outerbounds.com/engineering/operations/configure-metaflow/) using the
information in the CloudFormation stack outputs. The command below will launch an
interactive workflow and prompt you for the necessary variables:

```bash
metaflow configure aws --profile mlschool-aws
```

The command above creates a named profile named `mlschool-aws`. To keep using Metaflow
in *local* mode, create a file `~/.metaflowconfig/config_local.json` with an empty JSON
object in it. You can check
[https://docs.outerbounds.com/use-multiple-metaflow-configs/](https://docs.outerbounds.com/use-multiple-metaflow-configs/)
for more information about this:

```bash
echo '{}' > ~/.metaflowconfig/config_local.json
```

You can now enable the profile you want to use when running the pipelines by exporting
the `METAFLOW_PROFILE` variable in your local session. Por example, to run the pipelines
in *shared* mode, you can set the environment variable to `mlschool-aws`.

```bash
export METAFLOW_PROFILE=mlschool-aws
```

You can also prepend the profile name to a Metaflow command. For example, to run the
training pipeline in *local* mode, you can use the following command:

```bash
METAFLOW_PROFILE=local python3 pipelines/training.py --environment=pypi run
```

#### Running the Training pipeline

You can now run the Training pipeline using AWS Batch as the Compute Cluster by using
the `--with batch` parameter. This parameter will mark every step of the flow with the
`batch` decorator, which will instruct Metaflow to run the steps in AWS Batch:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/training.py \
    --environment=pypi run --with batch
```

While every step of the flow is running in a remote Compute Cluster, we are still using
the local environment to orchestrate the flow. Metaflow can use AWS Step Functions as
the Production Scheduler to orchestrate and schedule workflows. Check [Scheduling
Metaflow Flows with AWS Step
Functions](https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-aws-step-functions)
for more information.

Run the command below to deploy a version of the Training pipeline to AWS Step
Functions. This command will take a snapshot of your code, as well as the version of
Metaflow and export the whole package to AWS Step Functions for scheduling:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/training.py \
    --environment=pypi step-functions create
```

After the command finishes running, you should be able to list the existing state
machines in your account using the command below. You can also open the "Step Functions"
service in the AWS console to find the new state machine:

```bash
aws stepfunctions list-state-machines
```

To trigger the Training pipeline, you can use the `step-functions trigger` parameter
when running the flow. This command will create a new execution of the state machine:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/training.py \
    --environment=pypi step-functions trigger
```

#### Running the Deployment pipeline

To run the Deployment pipeline in the remote Compute Cluster, we need to modify the
permissions associated with one of the roles that we created with the Metaflow
CloudFormation stack. The new permissions will allow the role to access the Elastic
Container Registry (ECR) and to assume the role with the correct permissions to deploy
the model in SageMaker.

The role we need to modify has a name in the format
`[STACK_NAME]-BatchS3TaskRole-[SUFFIX]`. Assuming you named the Metaflow CloudFormation
stack, `metaflow`, you can retrieve the name of the role into a `$role` variable with
the following command:

```bash
role=$(aws iam list-roles --query 'Roles[*].RoleName' | grep metaflow-BatchS3TaskRole | tr -d '", ')
```

You can print the value of the `$role` variable to display the name of the role:

```bash
echo $role
```

We can now add the necessary permissions to deploy the model in SageMaker to the role
using the following command:

```bash
aws iam put-role-policy --role-name $role --policy-name mlschool \
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

At this point, you can run the Deployment pipeline in the remote Compute Cluster using
the following command:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/deployment.py \
    --environment=pypi run --with batch \
    --target sagemaker \
    --region $AWS_REGION \
    --assume-role $AWS_ROLE
```

Notice you need to specify the role using the `--assume-role` parameter when running the
pipeline. This role has permissions to create the necessary resources to host the model
in SageMaker.

To deploy the Deployment pipeline to AWS Step Functions, you can use the `step-functions
create` parameter:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/deployment.py \
    --environment=pypi step-functions create
```

To trigger the Deployment pipeline state machine, you can use the `step-functions
trigger` parameter:

```bash
METAFLOW_PROFILE=mlschool-aws python3 pipelines/deployment.py \
    --environment=pypi step-functions trigger \
    --target sagemaker \
    --region $AWS_REGION \
    --role $AWS_ROLE
```
