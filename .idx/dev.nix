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
      parseDotEnv = file:
        builtins.listToAttrs (
          map
            (line:
              let
                parts = builtins.split "=" line;
                key = builtins.elemAt parts 0;
                value = builtins.elemAt parts 1;
              in {
                name = key;
                value = value;
              }
            )
            (builtins.filter
              (l: l != "" && builtins.match " *#" l == null)
              (builtins.split "\n" (builtins.readFile file))
            )
        );

      sharedEnv = parseDotEnv ./env.shared;
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
