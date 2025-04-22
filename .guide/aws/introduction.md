# Production Pipelines In Amazon Web Services

In this section, we'll use Amazon Web Services (AWS) to run a remote MLflow server, run and orchestrate the pipelines in the cloud, and host the model.

Start by [installing the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). If you are running the project inside a Development Container, you should have the AWS CLI already installed.

The next step is to [create an AWS account](https://aws.amazon.com/free/) if you don't have one. We'll use this account to create the resources we need, so make sure you have administrator permissions. Configure an "Access Key ID" and "Secret Access Key" for this account, as we'll need these credentials going forward.

To configure your AWS account with the necessary resources, run the following command with the name of the user you want to create and the region where you want to deploy the resources. For example, this command will set up AWS with a user named `santiago` in the `us-east-1` region:

```shell
just aws-setup santiago us-east-1
```

This command will create a new CloudFormation stack based on the `cloud-formation/mlschool-cfn.yaml` template and configure your local environment with the necessary credentials. The command will use your access key and secret access key to create the stack.

The command will also modify the `.env` file in the repository's root directory to add the `AWS_USERNAME`, `AWS_ROLE`, `AWS_REGION`, and `BUCKET` environment variables. These variables are necessary to run the pipelines in the cloud.

When you finish using AWS, you can delete the stack and all the related resources by running the following command:

```shell
just aws-teardown us-east-1
```

If you are interested in learning how this command works, check out the [`scripts/aws.py`](scripts/aws.py) script.