import os
from pathlib import Path

import boto3
import click


@click.group()
def cli():
    """AWS Configuration CLI."""


@cli.command()
@click.option(
    "--stack-name", default="mlschool", help="Name of the CloudFormation stack"
)
@click.option(
    "--region", default="us-east-1", help="AWS region where the stack will be created"
)
@click.option("--user", help="Username for the new IAM user", required=True)
def setup(stack_name: str, region: str, user: str):
    """Set up AWS using a CloudFormation stack and configure the local environment.

    This command will set up the AWS account using a CloudFormation stack to prepare it
    to deploy the project. It will also configure the local environment with the
    necessary credentials and environment variables.

    Args:
        stack_name: The name of the CloudFormation stack that will be created
        region: The AWS region where the stack will be created
        user: The username for the new IAM user that will be created

    """
    try:
        session = _create_aws_session(region)

        # Load the CloudFormation template from disk.
        template_path = Path("cloud-formation/mlschool-cfn.yaml")
        if not template_path.exists():
            _error(
                f'CloudFormation template file "{template_path}" not found.')

        # Let's now create the CloudFormation stack in AWS.
        click.echo("Creating CloudFormation stack...")
        cf_client = session.client("cloudformation")
        cf_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_path.read_text(),
            Parameters=[
                {
                    "ParameterKey": "UserName",
                    "ParameterValue": user,
                },
            ],
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )

        # Wait for stack creation to finish before we move on.
        waiter = cf_client.get_waiter("stack_create_complete")
        waiter.wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 10, "MaxAttempts": 30},
        )
        click.echo("Stack creation finished.")

        # Get the stack outputs from the CloudFormation stack so we can use them
        # to configure the local environment.
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = {
            output["OutputKey"]: output["OutputValue"]
            for output in response["Stacks"][0]["Outputs"]
        }

        # The stack creates a new IAM user and a new IAM role. We need to get the
        # credentials for the new user. The secret access key associated with the
        # new user is stored in AWS Secrets Manager.
        sm_client = session.client("secretsmanager")
        secret_response = sm_client.get_secret_value(
            SecretId="/credentials/mlschool")

        _update_aws_credentials(
            username=outputs.get("User", ""),
            access_key_id=outputs.get("AccessKeyId", ""),
            secret_key=secret_response["SecretString"],
        )

        _update_env(
            variables_to_add={
                "AWS_USERNAME": outputs.get("User", ""),
                "AWS_ROLE": outputs.get("Role", ""),
                "AWS_REGION": outputs.get("Region", region),
                "BUCKET": outputs.get("Bucket", ""),
                "AWS_PROFILE": "mlschool",
            }
        )

        # Update AWS config with correct output keys
        _update_aws_config(
            username=outputs.get("User", ""),
            role_arn=outputs.get("Role", ""),
            region=outputs.get("Region", region),
        )

        click.echo("AWS configuration completed successfully.")

    except Exception as e:
        _error("Error configuring AWS", e)


@cli.command()
@click.option(
    "--stack-name", default="mlschool", help="Name of the CloudFormation stack"
)
@click.option(
    "--region", default="us-east-1", help="AWS region where the stack was created"
)
def teardown(stack_name: str, region: str):
    """Delete the CloudFormation stack and clean up AWS configuration.

    This command will delete the CloudFormation stack and clean up the AWS configuration
    on your local machine. It will also remove the environment variables from the .env
    file and the current session.

    Args:
        stack_name: The name of the CloudFormation stack that will be deleted
        region: The AWS region where the stack was created

    """
    # We want to make sure we remove the AWS_PROFILE environment variable so we
    # don't use the wrong profile when deleting the stack. We want boto3 to use
    # the default profile.
    os.environ.pop("AWS_PROFILE", None)

    session = _create_aws_session(region)
    cf_client = session.client("cloudformation")

    # Let's describe the existing stack to return the username associated with
    # the stack so we can remove it from the local AWS configuration files.
    try:
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = {
            output["OutputKey"]: output["OutputValue"]
            for output in response["Stacks"][0]["Outputs"]
        }
        username = outputs.get("User", "")
    except Exception:
        click.echo("Could not find existing stack info.")
        username = ""

    # We can now delete the stack.
    try:
        click.echo("Deleting CloudFormation stack...")
        cf_client.delete_stack(StackName=stack_name)

        # Let's wait for the deletion to complete before moving on.
        waiter = cf_client.get_waiter("stack_delete_complete")
        waiter.wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 10, "MaxAttempts": 30},
        )
        click.echo("Stack deletion completed!")
    except Exception as e:
        click.echo(f"Error deleting stack: {e}")
        click.echo("Continuing with local cleanup...")

    # Clean up AWS config file
    config_path = Path.home() / ".aws" / "config"
    updated_config = _remove_profiles_from_file(
        config_path, [f"profile {username}", "profile mlschool"]
    )
    config_path.write_text(updated_config)

    # Clean up AWS credentials file
    credentials_path = Path.home() / ".aws" / "credentials"
    updated_credentials = _remove_profiles_from_file(
        credentials_path, [username])
    credentials_path.write_text(updated_credentials)

    _update_env(variables_to_remove=[
        "AWS_USERNAME",
        "AWS_ROLE",
        "AWS_REGION",
        "BUCKET",
        "AWS_PROFILE",
    ])

    click.echo("AWS configuration cleanup completed successfully.")


def _remove_profiles_from_file(file_path: Path, profiles: list[str]) -> str:
    """Remove specified profiles from a file if they exist.

    Args:
        file_path: Path to the file to process
        profiles: List of profile names to remove

    Returns:
        str: File contents with specified profiles removed

    """
    if not file_path.exists():
        return ""

    file_content = file_path.read_text()
    lines = []
    skip = False
    for line in file_content.splitlines():
        if any(f"[{profile}]" in line for profile in profiles):
            skip = True
            continue
        if skip and line.strip() and not line.strip().startswith("["):
            continue
        if skip and (line.strip().startswith("[") or not line.strip()):
            skip = False
        if not skip:
            lines.append(line)

    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"

    return content


def _get_aws_session(
    access_key_id: str | None = None,
    secret_key: str | None = None,
    region: str | None = None,
) -> boto3.Session:
    """Return an AWS session using the provided credentials.

    Args:
        access_key_id: AWS Access Key ID
        secret_key: AWS Secret Access Key
        region: AWS region

    Returns:
        boto3.Session: Configured AWS session

    """
    if access_key_id and secret_key:
        return boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    return boto3.Session(region_name=region)


def _create_aws_session(region: str) -> boto3.Session:
    """Create an AWS session using default credentials or prompt for new ones.

    This function will first try to create an AWS session using the default credentials
    configured in the user's system. If no credentials are found, the user will be
    prompted to enter their AWS Access Key ID and AWS Secret Access Key. These
    credentials will be saved in the user's home directory so they can be reused later.

    Args:
        region: AWS region to use for the session

    Returns:
        boto3.Session: Configured AWS session

    """
    try:
        session = _get_aws_session(region=region)
        sts = session.client("sts")
        sts.get_caller_identity()
    except Exception:
        # If we can't find default credentials, prompt the user to enter
        # the AWS Access Key ID and AWS Secret Access Key.
        access_key_id = click.prompt("AWS Access Key ID", type=str)
        secret_key = click.prompt(
            "AWS Secret Access Key", type=str, hide_input=True)
        session = _get_aws_session(access_key_id, secret_key, region)

        # Update the AWS credentials file with the new credentials.
        _update_aws_credentials("default", access_key_id, secret_key)

    return session


def _update_aws_credentials(username: str, access_key_id: str, secret_key: str):
    """Update the AWS credentials file with the supplied credentials.

    Args:
        username: The username that will be used to access the resources
        access_key_id: AWS Access Key ID
        secret_key: AWS Secret Access Key

    """
    file_path = Path.home() / ".aws" / "credentials"
    file_path.parent.mkdir(exist_ok=True)

    # Prepare the configuration using the supplied credentials.
    credentials = f"[{username}]\n"
    credentials += f"aws_access_key_id = {access_key_id}\n"
    credentials += f"aws_secret_access_key = {secret_key}\n"

    # The credentials file might already have the new credentials we want to add, so
    # we need to remove the existing credentials first.
    content = _remove_profiles_from_file(file_path, [username])

    # Now we can add the new profile to the credentials file.
    file_path.write_text(content + credentials)


def _update_aws_config(username: str, role_arn: str, region: str):
    """Update the AWS config file with the required profiles.

    Args:
        username: The username that will be used to access the resources
        role_arn: The ARN of the role that will be used to access the resources
        region: The region where the resources are located

    """
    file_path = Path.home() / ".aws" / "config"
    file_path.parent.mkdir(exist_ok=True)

    # First, we need to create a profile that will be used to assume the role and
    # link it to the source profile.
    profile = (
        f"[profile mlschool]\n"
        f"role_arn = {role_arn}\n"
        f"source_profile = {username}\n"
        f"region = {region}\n"
    )

    # Next, we need to create the source profile using the supplied username.
    source_profile = f"[profile {username}]\nregion = {region}\n"

    # The config file might already have the new profiles we want to create, so we need
    # to remove them first.
    content = _remove_profiles_from_file(file_path, [username, "mlschool"])

    # Update the config file with the updated profiles.
    file_path.write_text(content + source_profile + profile)


def _get_relevant_env_lines(
    lines: list[str],
    variables_to_add: dict | None,
    variables_to_remove: list[str] | None,
) -> tuple[list[str], set[str]]:
    """Return any relevant lines from the environment file.

    This function will return any lines from the environment file that are relevant to
    the variables we want to add or remove. It will update the values of any existing
    variables that we want to add.

    Args:
        lines: The lines from the environment file
        variables_to_add: Dictionary of variables to add or update
        variables_to_remove: List of variable names to remove

    Returns:
        A tuple containing the updated lines and the variables that were updated

    """
    content = []
    updated_vars = set()

    for line in lines:
        if not line.strip() or (
            variables_to_remove
            and any(line.startswith(var + "=") for var in variables_to_remove)
        ):
            continue

        if variables_to_add and "=" in line:
            key = line.split("=")[0]
            if key in variables_to_add:
                content.append(f"{key}={variables_to_add[key]}")
                updated_vars.add(key)
                continue
        content.append(line)

    return content, updated_vars


def _update_env(
    variables_to_add: dict | None = None, variables_to_remove: list[str] | None = None
):
    """Update environment variables in the .env file and the current session.

    This function can add new variables, update existing ones, or remove specified
    variables from the .env file. It can also update the current session's environment
    variables.

    Args:
        variables_to_add: Dictionary of variables to add or update
        variables_to_remove: List of variable names to remove

    """
    env_file = Path(".env")

    # Let's start by updating the current session's environment variables.
    if variables_to_add:
        os.environ.update(variables_to_add)
    if variables_to_remove:
        [os.environ.pop(var, None) for var in variables_to_remove]

    # If the file doesn't exist, we need to create it and write any new variables
    # to it, in case there are any.
    if not env_file.exists():
        if variables_to_add is not None:
            env_file.write_text(
                "\n".join(f"{key}={value}" for key,
                          value in variables_to_add.items())
                + "\n"
            )
        return

    # Let's now process the existing file and return any variables we want to keep
    # and make sure we update their values if necessary.
    content, updated_vars = _get_relevant_env_lines(
        env_file.read_text().splitlines(),
        variables_to_add,
        variables_to_remove,
    )

    # If we have any new variables to add, and we haven't already used them to update
    # their values, we need to add them to the content.
    if variables_to_add:
        content.extend(
            f"{key}={value}"
            for key, value in variables_to_add.items()
            if key not in updated_vars
        )

    # If we have any content, we need to write it to the file. If we don't have any,
    # we need to delete the file.
    if content:
        env_file.write_text("\n".join(content) + "\n")
    else:
        env_file.unlink()


def _error(message: str, e: Exception | None = None) -> None:
    """Raise an error and exit the program.

    Args:
        message: The error message that will be displayed to the user
        e: The exception that occurred

    """
    if e:
        click.echo(f"{message}: {e!s}", err=True)
        raise click.Abort from e

    click.echo(f"{message}", err=True)
    raise click.Abort


if __name__ == "__main__":
    cli()
