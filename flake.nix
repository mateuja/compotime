{
  description = "Compotime";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  inputs.pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, poetry2nix, flake-utils, pre-commit-hooks }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./.;
          editablePackageSources = { compotime = ./compotime; };
          python = pkgs.python311;
          preferWheels = true;
          groups = [ "dev" "docs" ];
        };
      in {
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              lint-nix = {
                enable = true;
                name = "Run linters for flake.nix";
                entry = "nixfmt flake.nix";
                types = [ "nix" ];
              };
              lint-python = {
                enable = true;
                name = "Run linter for python files";
                entry = "ruff compotime tests";
                types = [ "python" ];
              };
              format = {
                enable = true;
                name = "Run formatters";
                entry = let
                  script = pkgs.writeShellScript "run_formatters.sh" ''
                    black .
                    ruff -s --fix --exit-zero .
                  '';
                in toString script;
                types = [ "python" ];
              };
              tests = {
                enable = true;
                name = "Run tests";
                entry = "pytest --cov=compotime --cov-report=xml -n auto";
                types = [ "python" ];
              };
              docs = {
                enable = true;
                name = "Build documentation";
                entry = let
                  script = pkgs.writeShellScript "build_docs.sh" ''
                    rm -rf docs/source/_autosummary docs/build/*
                    make -C ./docs html
                  '';
                in toString script;
              };
            };
          };
        };
        devShell = pkgs.mkShell {
          inherit (self.checks.${system}.pre-commit-check) shellHook;
          packages = [ poetryEnv pkgs.poetry pkgs.pandoc pkgs.nixfmt ];
        };
      });
}
