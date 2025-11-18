# Running A Remote MLflow Server

To configure a remote MLflow server, we'll use a CloudFormation template to set up a remote instance on AWS where we can run the server. This template will create a `t2.micro` [EC2](https://aws.amazon.com/ec2/) instance running Ubuntu. This tiny computer has one virtual CPU and 1 GiB of RAM. Amazon offers [750 hours of free usage](https://aws.amazon.com/free/) every month for this instance type, which should be enough for you to complete the program without incurring any charges. To create the stack, run the following command from the repository's root directory:

```shell
aws cloudformation create-stack \
    --stack-name mlflow \
    --template-body file://cloud-formation/mlflow-cfn.yaml
```

You can open the "CloudFormation" service in your AWS console to check the status of the stack. It will take a few minutes for the status to change from "CREATE_IN_PROGRESS" to "CREATE_COMPLETE". Once it finishes, run the following command to download the private key associated with the EC2 instance and save it as `mlschool.pem` in your local directory:

```shell
just aws-pem
```

At this point, you can open the "EC2" service in your AWS console, and go to the "Instances" page to find the new instance you'll be using to run the MLflow server. Wait for the instance to finish initializing, and run the following command to connect to it:

```shell
just aws-ssh
```

The EC2 instance comes prepared with everything you need to run the MLflow server. From the terminal connected to the remote instance, run the following command to start the server, binding it to the public IP address of the instance:

```shell
mlflow server --host 0.0.0.0 --port 5000
```

Once the server starts running, open a browser in your computer and navigate to the instance's public IP address on port 5000 to make sure MLflow is running correctly. You can find the public IP address associated to the EC2 instance with the following command:

```shell
echo $(aws cloudformation describe-stacks --stack-name mlflow \
    --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" \
    --output text)
```

Finally, modify the value of the `MLFLOW_TRACKING_URI` environment variable in the `.env` file inside your repository's root directory and point it to the remote instance. The following command will update the variable and export it in your current shell:

```shell
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

When you are done using the remote server, delete the CloudFormation stack to remove the instance and avoid unnecessary charges. Check the [Cleaning up AWS resources](.guide/amazon-web-services/cleaning-up.md) section for more information.