{ pkgs, aws_access_key_id ? "", aws_secret_access_key ? "", aws_region ? "", ... }: 
let
  default_aws_access_key_id = "YOUR_DEFAULT_ACCESS_KEY_ID";

  default_aws_secret_access_key = "YOUR_DEFAULT_SECRET_ACCESS_KEY";
  default_aws_region = "YOUR_DEFAULT_REGION";
  
  final_aws_access_key_id = if aws_access_key_id == "" then default_aws_access_key_id else aws_access_key_id;
  final_aws_secret_access_key = if aws_secret_access_key == "" then default_aws_secret_access_key else aws_secret_access_key;
  final_aws_region = if aws_region == "" then default_aws_region else aws_region;
in {
  channel = "stable-23.11";

  bootstrap = ''
    # Copy the folder containing the `idx-template` files to the final
    # project folder for the new workspace. ${./.} inserts the directory
    # of the checked-out Git folder containing this template.
    cp -rf ${./.} "$out"
    chmod -R +w "$out"

    ${if final_aws_access_key_id != "" && final_aws_secret_access_key != "" && final_aws_region != "" then ''
      cat << EOF >> "$out"/.idx/aws.nix
      {
        AWS_ACCESS_KEY_ID = "${final_aws_access_key_id}";
        AWS_SECRET_ACCESS_KEY = "${final_aws_secret_access_key}";
        AWS_REGION = "${final_aws_region}";
      }
      EOF
    '' else "echo "{}" >> \"$out\"/.idx/aws.nix"
    }

    # Remove the template files
    rm -rf "$out/.git" "$out/idx-template".{nix,json}
    rm -rf "$out/backup"
  '';
}
