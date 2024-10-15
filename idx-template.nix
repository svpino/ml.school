{ pkgs, aws_access_key_id ? "", aws_secret_access_key ? "", aws_region ? "us-east-1", ... }: {
  channel = "stable-23.11";

  bootstrap = ''
    # Copy the folder containing the `idx-template` files to the final
    # project folder for the new workspace. ${./.} inserts the directory
    # of the checked-out Git folder containing this template.
    cp -rf ${./.} "$out"

    cat << EOF >> "$out/.aws"
    [default]
    aws_access_key_id = ${aws_access_key_id}
    aws_secret_access_key = ${aws_secret_access_key}
    aws_region = ${aws_region}
    EOF

    # Set some permissions
    chmod -R +w "$out"

    # Remove the template files
    rm -rf "$out/.git" "$out/idx-template".{nix,json}
  '';
}
