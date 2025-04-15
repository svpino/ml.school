import os
from pathlib import Path

import boto3
import click


@click.group()
def cli():
    """AWS Configuration CLI for ML School"""


def write_to_env(key: str, value: str):
    """Append or update a key-value pair in .env file"""
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n")
        return

    lines = env_path.read_text().splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        elif line.strip():
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n")


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

    current_content = file_path.read_text()
    lines = []
    skip = False
    for line in current_content.splitlines():
        if any(f"[{profile}]" in line for profile in profiles):
            skip = True
            continue
        if skip and line.strip() and not line.strip().startswith("["):
            continue
        if skip and (line.strip().startswith("[") or not line.strip()):
            skip = False
        if not skip:
            lines.append(line)

    existing_content = "\n".join(lines)
    if existing_content and not existing_content.endswith("\n"):
        existing_content += "\n"
    return existing_content


def update_aws_config(username: str, role_arn: str, region: str):
    """Update AWS config file with new profile"""
    config_path = Path.home() / ".aws" / "config"
    config_path.parent.mkdir(exist_ok=True)

    # Create the mlschool profile configuration
    profile_content = f"""
[profile mlschool]
role_arn = {role_arn}
source_profile = {username}
region = {region}
"""

    # Create the source profile configuration
    source_profile = f"""
[profile {username}]
region = {region}
"""

    # Combine both configurations
    new_profile = source_profile + profile_content

    # Remove existing profiles if they exist
    existing_config = _remove_profiles_from_file(
        config_path, [username, "mlschool"])

    # Add new profiles at the end
    config_path.write_text(existing_config + new_profile)


def update_aws_credentials(profile: str, access_key_id: str, secret_key: str):
    """Update the AWS credentials file with the supplied profile.

    Args:
        profile: AWS profile name
        access_key_id: AWS Access Key ID
        secret_key: AWS Secret Access Key

    """
    credentials_path = Path.home() / ".aws" / "credentials"
    credentials_path.parent.mkdir(exist_ok=True)

    # Prepare the configuration using the supplied credentials.
    credentials = f"[{profile}]\n"
    credentials += f"aws_access_key_id = {access_key_id}\n"
    credentials += f"aws_secret_access_key = {secret_key}\n"

    # Remove existing profile if it exists
    existing_credentials = _remove_profiles_from_file(
        credentials_path, [profile])

    # Now we can add the new profile to the credentials file.
    credentials_path.write_text(existing_credentials + credentials)


def set_env_vars(env_vars: dict):
    """Set environment variables in current session"""
    for key, value in env_vars.items():
        os.environ[key] = value


def get_aws_session(
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


def create_aws_session(region: str) -> boto3.Session:
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
        session = get_aws_session(region=region)
        sts = session.client("sts")
        sts.get_caller_identity()
    except Exception:
        # If we can't find default credentials, prompt the user to enter
        # the AWS Access Key ID and AWS Secret Access Key.
        access_key_id = click.prompt("AWS Access Key ID", type=str)
        secret_key = click.prompt(
            "AWS Secret Access Key", type=str, hide_input=True)
        session = get_aws_session(access_key_id, secret_key, region)

        # Update the AWS credentials file with the new credentials.
        update_aws_credentials("default", access_key_id, secret_key)

    return session


def handle_error(message: str, e: Exception | None = None) -> None:
    if e:
        click.echo(f"{message}: {e!s}", err=True)
        raise click.Abort from e

    click.echo(f"{message}", err=True)
    raise click.Abort


@cli.command()
@click.option(
    "--stack-name",
    default="mlschool",
    help="Name of the CloudFormation stack",
)
@click.option(
    "--region",
    default="us-east-1",
    help="AWS region where the stack will be created",
)
@click.option("--user-id", help="Username for the new IAM user", required=True)
def configure(stack_name: str, region: str, user_id: str):
    """Configure AWS credentials and environment variables."""
    try:
        session = create_aws_session(region)

        # Read CloudFormation template
        template_path = Path("cloud-formation/mlschool-cfn.yaml")
        if not template_path.exists():
            handle_error("CloudFormation template file not found")

        template_body = template_path.read_text()

        # Create CloudFormation client
        cf_client = session.client("cloudformation")

        # Create CloudFormation stack
        click.echo("Creating CloudFormation stack...")
        cf_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=[
                {
                    "ParameterKey": "UserName",
                    "ParameterValue": user_id,
                },
            ],
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )

        # Wait for stack creation to complete
        click.echo("Waiting for stack creation to complete...")
        waiter = cf_client.get_waiter("stack_create_complete")
        waiter.wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 10, "MaxAttempts": 30},
        )
        click.echo("Stack creation completed!")

        # Get stack outputs
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = {
            output["OutputKey"]: output["OutputValue"]
            for output in response["Stacks"][0]["Outputs"]
        }

        # Get the secret access key from Secrets Manager
        sm_client = session.client("secretsmanager")
        secret_response = sm_client.get_secret_value(
            SecretId="/credentials/mlschool")
        new_secret_key = secret_response["SecretString"]

        # Update AWS credentials file with the new user
        update_aws_credentials(
            username=outputs.get("User", ""),
            access_key_id=outputs.get("AccessKeyId", ""),
            secret_key=new_secret_key,
        )

        # Update .env file and set environment variables
        env_vars = {
            "AWS_USERNAME": outputs.get("User", ""),
            "AWS_ROLE": outputs.get("Role", ""),
            "AWS_REGION": outputs.get("Region", region),
            "BUCKET": outputs.get("Bucket", ""),
            "AWS_PROFILE": "mlschool",
        }

        # Write to .env file
        for key, value in env_vars.items():
            write_to_env(key, value)

        # Set in current session
        set_env_vars(env_vars)

        # Update AWS config with correct output keys
        update_aws_config(
            username=outputs.get("User", ""),
            role_arn=outputs.get("Role", ""),
            region=outputs.get("Region", region),
        )

        click.echo("AWS configuration completed successfully!")

    except Exception as e:
        handle_error("Error configuring AWS", e)


@cli.command()
@click.option("--stack-name", default="mlschool", help="Name of the CloudFormation stack")
@click.option("--region", default="us-east-1", help="AWS region where the stack was created")
def delete(stack_name: str, region: str):
    """Delete the CloudFormation stack and clean up AWS configuration"""
    try:
        session = create_aws_session(region)

        # Create CloudFormation client
        cf_client = session.client("cloudformation")

        # Get stack info before deletion to know which user to remove from config
        try:
            response = cf_client.describe_stacks(StackName=stack_name)
            outputs = {
                output["OutputKey"]: output["OutputValue"]
                for output in response["Stacks"][0]["Outputs"]
            }
            username = outputs.get("User", "")
        except Exception:
            click.echo(
                "Could not find existing stack info. Will still attempt cleanup.")
            username = ""

        # Delete the CloudFormation stack
        click.echo("Deleting CloudFormation stack...")
        cf_client.delete_stack(StackName=stack_name)

        # Wait for stack deletion to complete
        click.echo("Waiting for stack deletion to complete...")
        waiter = cf_client.get_waiter("stack_delete_complete")
        waiter.wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 10, "MaxAttempts": 30},
        )
        click.echo("Stack deletion completed!")

        # Clean up AWS config file
        config_path = Path.home() / ".aws" / "config"
        if config_path.exists():
            current_config = config_path.read_text()
            lines = current_config.splitlines()
            filtered_lines = []
            skip = False
            for line in lines:
                if (f"[profile {username}]" in line or
                        "[profile mlschool]" in line):
                    skip = True
                    continue
                if skip and line.strip() and not line.strip().startswith("["):
                    continue
                if skip and (line.strip().startswith("[") or not line.strip()):
                    skip = False
                if not skip:
                    filtered_lines.append(line)

            config_path.write_text("\n".join(filtered_lines))

        # Clean up AWS credentials file
        credentials_path = Path.home() / ".aws" / "credentials"
        if credentials_path.exists() and username:
            current_credentials = credentials_path.read_text()
            lines = current_credentials.splitlines()
            filtered_lines = []
            skip = False
            for line in lines:
                if f"[{username}]" in line:
                    skip = True
                    continue
                if skip and line.strip() and not line.strip().startswith("["):
                    continue
                if skip and (line.strip().startswith("[") or not line.strip()):
                    skip = False
                if not skip:
                    filtered_lines.append(line)

            credentials_path.write_text("\n".join(filtered_lines))

        # Remove environment variables from .env file and current session
        env_vars_to_remove = ["AWS_USERNAME", "AWS_ROLE",
                              "AWS_REGION", "BUCKET", "AWS_PROFILE"]

        # Remove from .env file
        env_path = Path(".env")
        if env_path.exists():
            lines = env_path.read_text().splitlines()
            filtered_lines = []
            for line in lines:
                if not any(line.startswith(var + "=") for var in env_vars_to_remove):
                    filtered_lines.append(line)

            if filtered_lines:
                env_path.write_text("\n".join(filtered_lines) + "\n")
            else:
                env_path.unlink()  # Delete empty .env file

        # Remove from current session
        for var in env_vars_to_remove:
            os.environ.pop(var, None)  # Remove if exists, ignore if doesn't

        click.echo("AWS configuration cleanup completed successfully!")

    except Exception as e:
        handle_error("Error during cleanup", e)


if __name__ == "__main__":
    cli()
