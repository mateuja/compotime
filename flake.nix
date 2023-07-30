{
  description = "Compotime";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, poetry2nix, flake-utils }:
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
        devShell = pkgs.mkShell {
          packages = [ poetryEnv pkgs.poetry pkgs.pandoc pkgs.nixfmt ];
        };
      });
}
