# Deploying to AWS Managed Services

In this section, we'll use [AWS Batch](https://aws.amazon.com/batch/) to run the pipelines and [AWS Step Functions](https://aws.amazon.com/step-functions/) to orchestrate them. Since these services are fully managed by AWS, they will require little maintenance and will be reliable and highly available.

We can run Metaflow pipelines in both _local_ and _shared_ modes. While the _local_ mode is ideal for developing and testing pipelines, the _shared_ mode is designed to run them in a [production environment](https://docs.metaflow.org/production/introduction).

In _shared_ mode, the Metaflow Development Environment and the Production Scheduler rely on a separate compute cluster to provision compute resources on the fly. A central Metadata Service will track all executions, and their results will be stored in a common Datastore. Check the [Metaflow Service Architecture](https://outerbounds.com/engineering/service-architecture/) for more information.

We can run the pipelines in _shared_ mode using AWS Batch as the Compute Cluster and AWS Step Functions as the Production Scheduler. Check [Using AWS
Batch](https://docs.metaflow.org/scaling/remote-tasks/aws-batch) for useful tips and tricks for running Metaflow on AWS Batch.

To get started, create a new CloudFormation stack named `metaflow` by following the [AWS Managed with CloudFormation](https://outerbounds.com/engineering/deployment/aws-managed/cloudformation/) instructions.

After the Cloud Formation stack is created, you can [configure the Metaflow client](https://outerbounds.com/engineering/operations/configure-metaflow/) using the information from the CloudFormation stack outputs. The command below will configure Metaflow with a profile named `production` using the appropriate configuration:

```shell
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

To keep using Metaflow in _local_ mode, you must run the following command to create a new profile with an empty configuration. You can check [Use Multiple Metaflow Configuration Files](https://docs.outerbounds.com/use-multiple-metaflow-configs/) for more information:

```shell
echo '{}' > ~/.metaflowconfig/config_local.json
```

You can now enable the profile you want to use when running the pipelines by exporting the `METAFLOW_PROFILE` variable in your local environment. For example, to run your pipelines in _shared_ mode, you can set the environment variable to `production`:

```shell
export METAFLOW_PROFILE=production
```

Remember to delete the `metaflow` CloudFormation stack as soon as you are done using it to avoid unnecessary charges. Check the [Cleaning up AWS resources](.guide/aws/cleaning-up.md) section for more information.

## Running the Training pipeline remotely

You can now run the Training pipeline remotely by using the `--with batch` and `--with retry` parameters. These will mark every step of the flow with the `batch` and `retry` decorators, They will instruct Metaflow to run every step in AWS Batch and retry them if they fail:

```shell
METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
    --environment conda run \
    --with batch \
    --with retry
```

You can also run the pipeline using the `just` command with the `aws-train` recipe:

```shell
just aws-train
```

At this point, the pipeline will run in a remote compute cluster but it will still use the local environment to orchestrate the workflow. You can [schedule the pipeline using AWS Step Functions](https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-aws-step-functions) using the command below:

```shell
just aws-train-sfn-create
```

The above command will take a snapshot of the pipeline code and deploy it to AWS Step Functions. After you run the command, list the existing state machines in your account and you'll see a new state machine associated with the Training pipeline:

```shell
aws stepfunctions list-state-machines
```

To trigger the state machine corresponding to the Training pipeline, use the `just` command with the `aws-train-sfn-trigger` recipe:

```shell
just aws-train-sfn-trigger
```

The above command will create a new execution of the state machine and run the Training pipeline in the remote compute cluster. You can check the status of the execution under the Step Functions service in your AWS console or by running the following command:

```shell
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

## Running the Deployment pipeline remotely

To run the Deployment pipeline in the remote compute cluster, you need to modify the permissions associated with one of the roles created by the `metaflow` CloudFormation stack. The new permissions will allow the role to access the Elastic Container Registry (ECR) and deploy the model to Sagemaker:

```shell
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

At this point, you can run the Deployment pipeline in the remote compute cluster using `aws-deploy` recipe:

```shell
just aws-deploy
```

To deploy the Deployment pipeline to AWS Step Functions, you can use the `just` command with the `aws-deploy-sfn-create` recipe:

```shell
just aws-deploy-sfn-create
```

To trigger the state machine corresponding to the Deployment pipeline, use the `aws-deploy-sfn-trigger` recipe:

```shell
just aws-deploy-sfn-trigger
```

Finally, you can check the status of the execution by running the command below:

```shell
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
