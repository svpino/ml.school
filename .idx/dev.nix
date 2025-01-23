# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-23.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.awscli2
    pkgs.sqlite
    pkgs.openssh
    pkgs.just
  ];

  env = {
    MAMBA_ROOT_PREFIX = "/run/micromamba";
    # METAFLOW_DATASTORE_SYSROOT_LOCAL = "/run/.metaflow";
    # METAFLOW_CARD_LOCALROOT = "/run/.metaflow/mf.cards";
    KERAS_BACKEND = "jax";
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
      "tideily.mlschool"
    ];

    workspace = {
      onCreate = {
        python-venv = ''
          python3 -m venv .venv
          source .venv/bin/activate
          pip install -U pip && pip install -r requirements.txt
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
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";
        mlflow-server = ''
          source .venv/bin/activate
          mlflow server -h 127.0.0.1 -p 5000 --backend-store-uri sqlite:///mlflow.db
        '';
      };
    };
  };
}
