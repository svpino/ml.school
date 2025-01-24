{ pkgs, ... }: {
  channel = "stable-24.11";

  bootstrap = ''
    # Copy the folder containing the `idx-template` files to the final
    # project folder for the new workspace. ${./.} inserts the directory
    # of the checked-out Git folder containing this template.
    cp -rf ${./.} "$out"
    chmod -R +w "$out"

    # Remove the template files
    rm -rf "$out/idx-template".{nix,json}
  '';
}
