{ pkgs, aws_access_key_id ? "", aws_secret_access_key ? "", aws_region ? "", ... }: {
  channel = "stable-23.11";

  bootstrap = ''
    # Copy the folder containing the `idx-template` files to the final
    # project folder for the new workspace. ${./.} inserts the directory
    # of the checked-out Git folder containing this template.
    cp -rf ${./.} "$out"
    chmod -R +w "$out"

    ${if aws_access_key_id != "" && aws_secret_access_key != "" && aws_region != "" then ''
      cat << EOF >> "$out"/.idx/aws.nix
      {
        AWS_ACCESS_KEY_ID = "${aws_access_key_id}";
        AWS_SECRET_ACCESS_KEY = "${aws_secret_access_key}";
        AWS_REGION = "${aws_region}";
      }
      EOF
    '' else ''"echo "{}" >> \"$out\"/.idx/aws.nix"''
    }

    # Remove the template files
    rm -rf "$out/.git" "$out/idx-template".{nix,json}
    rm -rf "$out/backup"
  '';
}
