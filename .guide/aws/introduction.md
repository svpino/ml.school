# Production Pipelines In Amazon Web Services

In this section, we'll use Amazon Web Services (AWS) to run a remote MLflow server, run and orchestrate the pipelines in the cloud, and host the model.

Start by [creating a new AWS account](https://aws.amazon.com/free/) if you don't have one.

After creating the account, navigate to the "CloudFormation" service in your AWS console, click on the "Create stack" drop-down button at top-right, and select "With new resources (standard)" option. On the "Specify template" section, upload the `cloud-formation/mlschool-cfn.yaml` template file and click the "Next" button.

Name the stack `mlschool`, specify a User ID that doesn't exist in your account, and follow the prompts to create the stack. CloudFormation will create this new user and add it to your list of IAM users. When you run the command, please wait a couple of minutes for the stack creation status to appear on your AWS CloudFormation dashboard. After a few minutes, the stack status will change to "CREATE_COMPLETE," and you can open the "Outputs" tab to access the output values you'll need during the next steps.

Modify the `.env` file in the repository's root directory to add the `AWS_USERNAME`, `AWS_ROLE`, `AWS_REGION`, and `BUCKET` environment variables. Before running the command below, replace the values within square brackets using the outputs from the CloudFormation stack:

```shell
export $( (tee -a .env << EOF
AWS_USERNAME=[AWS_USERNAME]
AWS_ROLE=[AWS_ROLE]
AWS_REGION=[AWS_REGION]
BUCKET=[BUCKET]
EOF
) && cat .env | xargs)
```

[Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your environment and configure it using the command below:

```shell
aws configure --profile $AWS_USERNAME
```

The configuration tool will ask for the "Access Key ID", "Secret Access Key", and "Region." You can get the "Access Key ID" and "Region" from the CloudFormation stack "Outputs" tab. To get the "Secret Access Key", navigate to the "Secrets Manager" service in your AWS console and retrieve the secret value under the `/credentials/mlschool` key.

Finally, configure the command line interface to use the role created by the CloudFormation template. Run the following command to update your local AWS configuration:

```shell
cat << EOF >> ~/.aws/config

[profile mlschool]
role_arn = $AWS_ROLE
source_profile = $AWS_USERNAME
region = $AWS_REGION
EOF
```

At this point, you should be able to take advantage of the role's permissions at the command line by using the `--profile` option on every AWS command, or you can export the `AWS_PROFILE` environment variable for the current session to make it the default profile:

```shell
export AWS_PROFILE=mlschool
```

To ensure the permissions are correctly set, run the following command to return a list of S3 buckets in your account:

```shell
aws s3 ls
```