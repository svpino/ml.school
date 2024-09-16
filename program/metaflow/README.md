
```
python3 training.py --environment=pypi card server
```


Ubuntu:

```
$ sudo apt-get update
$ sudo apt-get install build-essential python3.12-venv zlib1g-dev unzip

DOCKER
$ sudo usermod -a -G docker $USER
$ newgrp docker
```


```
$ mlflow deployments create -t sagemaker --name penguins -m models:/penguins/1 -C region_name=us-east-1 -C instance-type=ml.m4.xlarge -C instance-count=1
```


-------- METAFLOW INSTALL

Follow the [AWS Managed with CloudFormation](https://outerbounds.com/engineering/deployment/aws-managed/cloudformation/) instructions to deploy a CloudFormation template that will set up all the resources needed to enable cloud-scaling in Metaflow.

Fetch the API Gateway Key ID for the Metadata Service using the command below. Make sure to replace `[ApiKeyID]` with the value of the `ApiKeyID` output in the CloudFormation stack. You'll need this value to configure the `METAFLOW_SERVICE_AUTH_KEY` variable in the next step.

```bash
$ aws apigateway get-api-key --api-key [ApiKeyID] --include-value | grep value
```

[Configure the Metaflow client](https://outerbounds.com/engineering/operations/configure-metaflow/) with the information in the CloudFormation stack outputs. The command below will launch an interactive workflow and prompt you for the necessary variables:

```bash
$ metaflow configure aws --profile aws-batch
```

You can enable the Metaflow profile by exporting the `METAFLOW_PROFILE` variable to your environment:

```bash
$ export METAFLOW_PROFILE=aws-batch
```

--------



## Preparing Your Local Environment

TBD

The code in the program runs on any Unix-based operating system (e.g. Ubuntu or MacOS). If you are using Windows, install the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about) (WSL) and you'll be able to follow the instructions below and run the code without issues.

Start by forking the program's [GitHub Repository](https://github.com/svpino/ml.school) and clonning it on your local computer. This will allow you to modify your copy of the code and push the changes to your personal repository.

The code in the repository was written using Python 3.12, so make sure you have the same [Python version](https://www.python.org/downloads/release/python-3126/) installed in your environment. A more recent version of Python should work as well, but sticking to the same version will avoid any potential issues.

After cloning the repository, navigate to the main directory and create and activate a virtual environment. We'll install all the required libraries inside this virtual environment. This will prevent any conflicts with other projects you might have on your computer:

```bash
$ python3 -m venv .venv 
$ source .venv/bin/activate
```

Now, within the virtual environment, you can update `pip` and install the libraries specified in the `requirements.txt` file:

```bash
$ pip3 install -U pip
$ pip3 install -r requirements.txt
```

At this point, you should have a working Python environment with all the required dependencies. The final step is to create an `.env` file inside the repository's main directory. We'll use this file to define the environment variables we need to run the code in the repository:

```bash
$ echo "KERAS_BACKEND=jax" >> .env
```

The command above will create the `.env` file with the `KERAS_BACKEND` environment variable. This variable specifies the framework we want Keras to use when training a model.

Finally, install Docker using the [installation instructions](https://docs.docker.com/engine/install/) for your particular environment. After you install it, you can verify it's running using the following command:

```bash
$ docker ps
```

## Setting Up Amazon Web Services

We'll use Amazon Web Services (AWS) at different points in the program to run the pipelines in the cloud, host the model, and run a remote MLflow server. 

Start by [creating a new AWS account](https://aws.amazon.com/free/) if you don't have one.

After you create an account, navigate to the *CloudFormation* service in your AWS console, click on the *Create stack* button and select *With new resources (standard)*. On the *Specify template* section, upload the `cloud-formation/mlschool-cfn.yaml` template file and click on the *Next* button. Specify a name for the stack and a name for a user account and follow the prompts to create the stack. After a few minutes, the stack status will change to *CREATE_COMPLETE* and you'll be able to open the *Outputs* tab to access the output values you'll need for the next steps.

After creating the stack, [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment and configure it using the command below. Replace `[AWS USERNAME]` with the name of the user you specified when creating the CloudFormation stack:

```bash
$ aws configure --profile [AWS USERNAME]
```

The configuration tool will ask for the **Access Key ID** and **Secret Access Key** associated to the user. You can get the **Access Key ID** from the CloudFormation stack *Outputs* tab. To get the **Secret Access Key**, navigate to the *Secrets Manager* service in your AWS console and retrieve the secret value related to the `/credentials/mlschool` key. The configuration tool will also ask for the **Region** you'll be using. You can get this value from the CloudFormation stack *Outputs* tab. 

We need to configure the command line interface to use the role created by the CloudFormation template. Open the `~/.aws/config` file in your favorite text editor and append the lines below. Replace `[AWS ROLE ARN]`, `[AWS USERNAME]`, and `[AWS REGION]` with the appropriate values:

```bash
[profile mlschool]
role_arn = [AWS ROLE ARN]
source_profile = [AWS USERNAME]
region = [AWS REGION]
```

At this point, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command. For example, the following command should return a list of S3 buckets in your account:

```bash
$ aws s3 ls --profile mlschool
```

To take advantage of the role's permissions on every command without having to specify the `--profile` option, export the `AWS_PROFILE` environment variable for the current session:

```bash
$ export AWS_PROFILE=mlschool
```

You can verify that the `mlschool` profile is being used by running the following command and looking at the `Arn` associated to your identity. You want this value to correspond to the role we created using the CloudFormation template:

```bash
$ aws sts get-caller-identity
```

## Setting Up an MLflow Server

MLflow is a platform-agnostic machine learning lifecycle management tool that will help us track experiments and share and deploy models. For this program, you can run the MLflow server locally to keep everything in your local computer. For a more scalable solution, you should run MLflow from a remote server.

* [Running the MLflow Server Locally](#running-the-mlflow-server-locally)
* [Running the MLflow Server in a Remote Server](#running-the-mlflow-server-in-a-remote-server)

### Running the MLflow Server Locally

Open a terminal window, activate the virtual environment you created previously, and install the `mlflow` library using the command below:

```bash
$ pip3 install mlflow
```

Once you install the `mlflow` library, you can run the server with the following command:

```bash
$ mlflow server --host 127.0.0.1 --port 5000
```

Once running, you can open [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your web browser to see the user interface.

By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database to store the tracking data using the parameter `--backend-store-uri`:

```bash
$ mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db
```

For more information on the MLflow server, run the following command:

```bash
$ mlflow server --help
```

After the server is running, modify the `.env` file inside the repository's main directory and add the `MLFLOW_TRACKING_URI` environment variable pointing to the tracking URI of the MLflow server:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

You can now export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

### Running the MLflow Server in a Remote Server

To configure a remote MLflow server, we'll use a CloudFormation template to set up a remote instance on AWS where we'll run the server. This template will create a `t2.micro` EC2 instance running Ubuntu. This is a very small computer, with 1 virtual CPU and 1 GiB of RAM. Amazon offers 750 hours of free usage every month for this instance type, which should be enough for you to complete the program without incurring any charges.

To create the stack, run the following command from the repository's main directory:

```bash
$ aws cloudformation create-stack \
    --stack-name mlschool-mlflow \
    --template-body file://cloud-formation/mlflow-cfn.yaml
```

You can open the *CloudFormation* service in your AWS console to check the status of the stack. It will take a few minutes for the status to change from `CREATE_IN_PROGRESS` to `CREATE_COMPLETE`. Once it finishes, open the *Outputs* tab, copy the value of the **KeyPair** output and use it to replace `[KEY PAIR]` in the command below. This command will download the private key associated with the EC2 instance and save it as `mlschool.pem` in your local directory:

```bash
$ aws ssm get-parameters --names "/ec2/keypair/[KEY PAIR]" \
    --with-decryption | python3 -c 'import json;import sys;o=json.load(sys.stdin);print(o["Parameters"][0]["Value"]);' \
    > mlschool.pem
```

Change the permissions on the private key file to ensure the file is not publicly accessible:

```bash
$ chmod 400 mlschool.pem
```

At this point, you can open the *EC2* service, and go to the *Instances* page to find the new instance you'll be using to run the MLflow server. Wait for the instance to finish initializing, select it, click on the *Connect* button, and open the *SSH client* tab to see the instructions on how to connect to it.

Open a terminal window and run the `ssh` command suggested in the connection instructions. It should look something like this, where `[EC2 PUBLIC DNS]` is the public DNS name of the EC2 instance:

```bash
$ ssh -i "mlschool.pem" ubuntu@[EC2 PUBLIC DNS]
```

Once inside the instance, run the following command to display the public IP address assigned to the instance. You can also find this value in the EC2 console. You'll use this value to connect to the MLflow server from your local environment:

```bash
$ curl ifconfig.me
```

The EC2 instance comes prepared with everything you need to run the MLflow server, so you can run the following command to start and bind the server to the public IP address of the instance:

```bash
$ mlflow server --host 0.0.0.0 --port 5000
```

Once running, open the `http://[EC2 PUBLIC IP]:5000` URL in your web browser. Replace `[EC2 PUBLIC IP]` with the public IP address assigned to the EC2 instance.

At this point, you can modify the `.env` file inside the repository's main directory to add the `MLFLOW_TRACKING_URI` environment variable. This variable should point to the URL of the MLflow server:

```bash
MLFLOW_TRACKING_URI=http://[EC2 PUBLIC IP]:5000
```

You can now export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

When you are done using the MLflow server, delete the CloudFormation stack to avoid unnecessary charges:

```bash
$ aws cloudformation delete-stack --stack-name mlschool-mlflow
```

## Running The Training Pipeline

The training pipeline trains, evaluates, and registers a model in the MLflow model registry.

Run the training pipeline locally using the following command:

```bash
$ python3 training.py --environment=pypi run
```

After the pipeline finishes, you should see a new version of the model in the MLflow model registry.

If you want to run the training pipeline on AWS Batch, make sure you follow the [Distributed Pipelines using AWS Managed Services](#distributed-pipelines-using-aws-managed-services) instructions to setup your AWS account. Then, you can use the following command: 

```bash
$ python3 training.py --environment=pypi --datastore=s3 run --with batch
```

For more information on the training pipeline and the parameters you can use to customize it, you can run the following command:

```bash
$ python3 training.py --environment=pypi run --help
```

## Distributed Pipelines using AWS Managed Services

We can run the metaflow pipelines in a *shared mode* using AWS Batch for compute and AWS Step Functions for orchestrating workflows in production. 

The Development Environment and the Production Scheduler rely on a separate Compute Cluster to provision compute resources on the fly. All executions are tracked by a central Metadata Service and their results are persisted in a common Datastore. Check [Service Architecture](https://outerbounds.com/engineering/service-architecture/) for more information on the Metaflow architecture.

The following diagram highlights the services used by Metaflow and their role in the Metaflow stack:

![Metaflow Architecture](https://outerbounds.com/assets/images/service-arch-02-514547c94fd621e418cacd085f5b4f61.png)


1. You must be logged onto AWS as an account with sufficient permissions to provision the required resources. A simple way to get the necessary permissions is to add the user to a group with the `AdministratorAccess` policy. Keep in mind that, while convenient, this approach is not recommended for production environments.

2. Follow the [CloudFormation instructions](https://outerbounds.com/engineering/deployment/aws-managed/cloudformation/) to download the CloudFormation template and create the stack.

3. Modify the `.env` file inside the repository's main directory with the environment variables below. You'll find the values for the variables in the *Outputs* tab of the CloudFormation stack you created in the previous step:

```bash
METAFLOW_SERVICE_AUTH_KEY=[METAFLOW SERVICE AUTH KEY]
METAFLOW_BATCH_JOB_QUEUE=[METAFLOW BATCH JOB QUEUE]
METAFLOW_SFN_DYNAMO_DB_TABLE=[METAFLOW SFN DYNAMO DB TABLE]
METAFLOW_ECS_S3_ACCESS_IAM_ROLE=[METAFLOW ECS S3 ACCESS IAM ROLE]
METAFLOW_EVENTS_SFN_ACCESS_IAM_ROLE=[METAFLOW EVENTS SFN ACCESS IAM ROLE]
METAFLOW_SERVICE_INTERNAL_URL=[METAFLOW SERVICE INTERNAL URL]
METAFLOW_DATASTORE_SYSROOT_S3=[METAFLOW DATASTORE SYSROOT S3]
METAFLOW_DATATOOLS_S3ROOT=[METAFLOW DATATOOLS S3ROOT]
METAFLOW_SERVICE_URL=[METAFLOW SERVICE URL]
METAFLOW_SFN_IAM_ROLE=[METAFLOW SFN IAM ROLE]
```






4. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

You should delete the CloudFormation stack as soon as you are done using it to avoid unnecessary charges.


AWS Step Functions

This command takes a snapshot of your code in the working directory, as well as the version of Metaflow used and exports the whole package to AWS Step Functions for scheduling:

```bash
$ python3 training.py --environment=pypi --datastore=s3 --with retry step-functions create
```




## Deploying the Model

You can deploy the model to a variety of deployment targets. Except when deploying the model locally, you'll run the feployment pipeline specifying the target platform using the `--target` parameter. This pipeline will connect to the platform, create a new endpoint to host the model, and run a few samples to test that everything is working as expected.

For specific information on each supported deployment target, follow the respective links below:

* [Deploying the model as a local inference server](#deploying-the-model-as-a-local-inference-server)
* [Deploying the model to SageMaker](#deploying-the-model-to-sagemaker)
* [Deploying the model to Azure Machine Learning](#deploying-the-model-to-azure-machine-learning)

### Deploying the model as a local inference server

To deploy your model locally, you can use the `mflow models serve` command and specify the model version you want to deploy from the model registry. This command will use the active Python environment to execute the model.

This command starts a local server listening on the specified port and network interface. Make sure you replace `[MODEL VERSION]` with the version of the model you want to deploy:

```bash
$ mlflow models serve -m models:/penguins/[MODEL VERSION] \
    -h 0.0.0.0 -p 8080 \
    --no-conda
```

You can now test the model by sending a request to the server. The following command should return a prediction for the provided input:

```bash
$ curl -X POST http://0.0.0.0:8080/invocations \
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

You can find more information about local deployments in [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

### Deploying the model to SageMaker

1. [Create a new AWS account](https://aws.amazon.com/free/) if you don't have one.

2. To deploy the model in SageMaker, we'll need access to `ml.m4.xlarge` instances. By default, the quota for most new accounts is zero, so you'll need to request a quota increase. You can do this in your AWS account under *Service Quotas* > *AWS Services* > *Amazon SageMaker*. Find `ml.m4.xlarge for endpoint usage` and request a quota increase of 8 instances.

3. Go to the *IAM service* and create a new role with the name `penguins`. We'll use this role to define the permissions we need to run the deployment pipeline.

4. After creating the role, modify its trust relationship to allow any user to assume this role. Open the *Trust relationships* tab and modify its content using the document below. Make sure you replace `[AWS ACCOUNT ID]` with your AWS account ID:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::[AWS ACCOUNT ID]:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {}
        }
    ]
}
```

5. Open the *Permissions* tab and create an inline policy for the `penguins` role using the document below. This policy will allow any user assuming the role to access the required resources to deploy the model in SageMaker:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": [
                "iam:PassRole",
                "iam:GetRole"
            ],
            "Resource": "*"
        },
        {
            "Sid": "SageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:ListEndpoints",
                "sagemaker:DescribeEndpoint",
                "sagemaker:CreateEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DescribeModel",
                "sagemaker:CreateModel",
                "sagemaker:DeleteEndpoint",
                "sagemaker:ListTags",
                "sagemaker:AddTags",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        },
        {
            "Sid": "ECR",
            "Effect": "Allow",
            "Action": [
                "ecr:DescribeRepositories",
                "ecr:CreateRepository",
                "ecr:GetAuthorizationToken",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:DeleteRepository",
                "ecr:PutImage"
            ],
            "Resource": "*"
        },
        {
            "Sid": "S3",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:ListBucket",
                "s3:ListAllMyBuckets",
                "s3:GetBucketLocation",
                "s3:PutObject",
                "s3:PutObjectTagging",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::*"
        }
    ]
}
```

6. Under the *IAM service*, create a new user account. We'll use this user to interact with the platform through the console and the command line interface.

7. Open the *Permissions* tab and create an inline policy for the user using the document below. This policy will allow the user to assume the `penguins` role. Make sure to replace `[AWS ACCOUNT ID]` with your AWS account ID.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::[AWS ACCOUNT ID]:role/penguins"
        }
    ]
}
```

8. Open the *Security Credentials* tab and create an *access key*. Write down the **Access Key ID** and **Secret Access Key** values so you can use them later.

9. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment.

10. Configure the AWS CLI using the following command. You will need to specify your **Access Key**, **Secret Access Key**, and the **region** where you'll be deploying the model. Make sure to replace `[AWS USERNAME]` with the name of the user you created before:

```bash
$ aws configure --profile [AWS USERNAME]
```

11. We need to configure the command line interface to use the role we created using the Cloud Formation template. Append the following lines to the `~/.aws/config` file. Make sure to replace `[AWS ROLE ARN]`, `[AWS USERNAME]`, and `[AWS REGION]` with the appropriate values. After this, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command. 

```bash
[profile mlschool]
role_arn = [AWS ROLE ARN]
source_profile = [AWS USERNAME]
region = [AWS REGION]
```

12. Export the `AWS_PROFILE` environment variable for the current session. This will allow you to take advantage of the `mlschool` role's permissions on every command without having to specify the `--profile` option:

```bash
$ export AWS_PROFILE=mlschool
```

13. If it doesn't exist, create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` and `[AWS REGION]` with the appropriate values:

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

SAGEMAKER_REGION=[AWS REGION]
```

14. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

15. To host the model in SageMaker, we need to build a Docker image and push it to the Elastic Container Registry (ECR) in AWS. You can accomplish this by running the following command:

```bash
$ mlflow sagemaker build-and-push-container
```

#### Running the deployment pipeline

After you finish setting up your AWS account, you can run the deployment pipeline from the repository's main directory using the following command:

```bash
$ python3 deployment.py --environment=pypi \
    run --target sagemaker \
    --endpoint $ENDPOINT_NAME
```

As soon as you are done with the SageMaker endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
```



### Deploying the model to Azure Machine Learning

1. Create a [free Azure account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=azurefreeaccount) if you don't have one.

2. Install the Azure [Command Line Interface (CLI)](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and [configure it on your environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public). After finishing these steps, you should be able to run the following command to display your Azure subscription and configuration:

```bash
$ az account show && az configure -l
```

3. In the Azure Portal, find the *Resource providers* tab under your subscription. Register the `Microsoft.Cdn` and the `Microsoft.PolicyInsights` providers.

4. To deploy the model to an endpoint, we need to request a quota increase for the virtual machine we'll be using. In the Azure Portal, open the *Quotas* tab and filter the list by the *Machine learning* provider, your subscription, and your region. Request a quota increase for the `Standard DSv2 Family Cluster Dedicated vCPUs`. Set the new quota limit to 16.

5. If it doesn't exist, create an `.env` file inside the repository's main directory with the environment variables below. Make sure to replace `[MLFLOW URI]` and `[AZURE SUBSCRIPTION ID]` with the appropriate values:

```bash
MLFLOW_TRACKING_URI=[MLFLOW URI]
ENDPOINT_NAME=penguins

AZURE_SUBSCRIPTION_ID=[AZURE SUBSCRIPTION ID]
AZURE_RESOURCE_GROUP=mlschool
AZURE_WORKSPACE=main

```

6. Export the environment variables from the `.env` file in your current shell:

```bash
$ export $(cat .env | xargs)
```

#### Running the deployment pipeline

After you finish setting up your Azure account, you can run the deployment pipeline from the repository's main directory using the following command:

```bash
$ python3 deployment.py --environment=pypi \
    run --target azure \
    --endpoint $ENDPOINT_NAME
```

For more information on the deployment pipeline and the parameters you can use to customize it, you can run the following command:

```bash
$ python3 deployment.py --environment=pypi run --help
```

After you are done with the Azure endpoint, make sure you delete it to avoid unnecessary costs:

```bash
$ az ml online-endpoint delete --name $ENDPOINT_NAME \
    --resource-group $AZURE_RESOURCE_GROUP \
    --workspace-name $AZURE_WORKSPACE \
    --yes
```

You can also delete the entire resource group if you aren't planning to use it anymore. This will delete all the resources you created to host the model:

```bash
$ az group delete --name $AZURE_RESOURCE_GROUP
```