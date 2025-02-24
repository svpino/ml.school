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

  env = {
    MAMBA_ROOT_PREFIX = "/run/micromamba";
    # METAFLOW_DATASTORE_SYSROOT_LOCAL = "/run/.metaflow";
    # METAFLOW_CARD_LOCALROOT = "/run/.metaflow/mf.cards";
    KERAS_BACKEND = "tensorflow";
    ENDPOINT_NAME = "penguins";
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000";
    METAFLOW_PROFILE = "local";
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
