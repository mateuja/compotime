{
  description = "Compotime";
  
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, poetry2nix }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ poetry2nix.overlay ];
          };

          poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
            projectDir = ./.;
            editablePackageSources = {
              compotime = ./compotime;
            };
            python = pkgs.python311;
            preferWheels = true;
            groups = [ "dev" "docs" ];
          };
        in
        {
          default = pkgs.mkShell { packages = [ poetryEnv pkgs.poetry pkgs.pandoc ]; };
        });
    };
}
