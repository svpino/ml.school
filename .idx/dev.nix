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
    pkgs.azure-cli
  ];

  env = {
    KERAS_BACKEND = "jax";
    ENDPOINT_NAME = "penguins";
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000";
  };

  services.docker.enable = true;

  idx = {
    
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "charliermarsh.ruff"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # web = {
        #   # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
        #   # and show it in IDX's web preview panel
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     # Environment variables to set for your server
        #     PORT = "$PORT";
        #   };
        # };
      };
    };

    workspace = {
      onCreate = {
        python-venv = ''
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
        '';

        metaflow-config = ''
          mkdir ~/.metaflowconfig
          echo '{}' > ~/.metaflowconfig/config_local.json
        '';

        aws-config = "mv .aws ~/";
        
        default.openFiles = [ "README.md" ];
      };
      onStart = {
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";
        mlflow-server = ''
          source .venv/bin/activate
          mlflow server -h 127.0.0.1 -p 5000
        '';
      };
    };
  };
}
