# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.gcc
    pkgs.awscli2
    pkgs.sqlite
    pkgs.openssh
    pkgs.just
    pkgs.uv
  ];

  # Load shared environment variables
  env =
    let
      # A robust .env parser
      parseDotEnv = file:
        let
          content = builtins.readFile file;
          lines = builtins.split "\n" content;

          # Parses a single line into a { name, value } attribute set.
          parseLine = line:
            # Ignore comments and empty lines.
            if ! (builtins.isString line) || line == "" || (builtins.substring 0 1 line) == "#" then
              null
            else
              # Find the first '='.
              let
                match = builtins.match "([^=]+)=(.*)" line;
              in
              # If there's no '=', it's not a valid line.
              if match == null then
                null
              else
                {
                  name = builtins.elemAt match 0;
                  value = builtins.elemAt match 1;
                };
      
          # Process all lines and filter out the nulls (invalid lines).
          parsedLines = builtins.filter (x: x != null) (map parseLine lines);
        in
          # Convert the list of { name, value } pairs to an attribute set.
          builtins.listToAttrs parsedLines;

      sharedEnv = parseDotEnv ./../env.shared;
    in
      sharedEnv // {
        MAMBA_ROOT_PREFIX = "/run/micromamba";
        # METAFLOW_DATASTORE_SYSROOT_LOCAL = "/run/.metaflow";
        # METAFLOW_CARD_LOCALROOT = "/run/.metaflow/mf.cards";
        METAFLOW_PROFILE = "local";
        
        # We need to set the PYTHONPATH environment variable to the src directory
        # so Python can find the project modules.
        PYTHONPATH = "src";
      };

  services.docker.enable = true;

  idx = {
    
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "charliermarsh.ruff"
      "ms-toolsai.jupyter"
      "tideily.mlschool"
    ];

    workspace = {
      onCreate = {
        uv-sync = ''
          export CC=gcc
          uv sync
        '';

        environment = ''
          cat << EOF >> .env
          KERAS_BACKEND=$KERAS_BACKEND
          ENDPOINT_NAME=$ENDPOINT_NAME 
          MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
          EOF
        '';

        metaflow-config = ''
          mkdir -p ~/.metaflowconfig
          echo '{}' > ~/.metaflowconfig/config_local.json
        '';

        default.openFiles = [ "README.md" ];
      };
      onStart = {
      };
    };
  };
}
