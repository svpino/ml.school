{ pkgs, aws_access_key_id ? "", aws_secret_access_key ? "", aws_region ? "us-east-1", ... }: {
  channel = "stable-23.11";

  bootstrap = ''
    # Copy the folder containing the `idx-template` files to the final
    # project folder for the new workspace. ${./.} inserts the directory
    # of the checked-out Git folder containing this template.
    cp -rf ${./.} "$out"

    # Set some permissions
    chmod -R +w "$out"

    mkdir -p ~/.aws

    cat << EOF >> ~/.aws/credentials
    [default]
    aws_access_key_id = ${aws_access_key_id}
    aws_secret_access_key = ${aws_secret_access_key}
    EOF

    cat << EOF >> ~/.aws/config
    [default]
    region = ${aws_region}
    EOF

    # Remove the template files
    rm -rf "$out/idx-template".{nix,json}
  '';
}
