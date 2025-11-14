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

    # TensorFlow uses its own logging system, so let's set this environment
    # variable to suppress their logs.
    # 0=all logs, 1=no INFO, 2=no WARNING, 3=no ERROR
    TF_CPP_MIN_LOG_LEVEL = "2";

    KERAS_BACKEND = "tensorflow";
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000";
    ENDPOINT_NAME = "penguins";
    METAFLOW_PROFILE = "local";

    # We need to set the PYTHONPATH environment variable to the src directory
    # so Python can find the project modules.
    PYTHONPATH = "src";

    # We want to suppress MLflow's printing of logs to the terminal.
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT = "1";

    # Google ADK outputs a warning message whenever we use LiteLLM with a Gemini
    # model instead of using Gemini directly. We can suppress these warnings by
    # setting this environment variable.
    ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS = "true";
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
