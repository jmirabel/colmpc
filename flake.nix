{
  description = "Collision avoidance for MPC";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    nixpkgs.follows = "gepetto/nixpkgs";
    nix-ros-overlay.follows = "gepetto/nix-ros-overlay";
    treefmt-nix.follows = "gepetto/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      imports = [ inputs.treefmt-nix.flakeModule ];
      perSystem =
        {
          lib,
          pkgs,
          system,
          self',
          ...
        }:
        {
          # Drop this once crocoddyl >= 3.0.1 reaches nix-ros-overlay
          _module.args.pkgs =
            let
              pkgsForPatching = inputs.nixpkgs.legacyPackages.x86_64-linux;
              patchedNixpkgs = (
                pkgsForPatching.applyPatches {
                  inherit (inputs.gepetto) patches;
                  name = "patched nixpkgs";
                  src = inputs.nixpkgs;
                }
              );
            in
            import patchedNixpkgs {
              inherit system;
              overlays = [
                inputs.nix-ros-overlay.overlays.default
                inputs.gepetto.overlays.default
              ];
            };
          checks = lib.mapAttrs' (n: lib.nameValuePair "package-${n}") self'.packages;
          packages =
            let
              src = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions [
                  ./examples
                  ./include
                  ./python
                  ./tests
                  ./CMakeLists.txt
                  ./package.xml
                  ./pyproject.toml
                ];
              };
            in
            {
              default = self'.packages.py-colmpc;
              colmpc = pkgs.colmpc.overrideAttrs { inherit src; };
              py-colmpc = pkgs.python3Packages.colmpc.overrideAttrs {
                inherit src;
              };
            };
          treefmt.programs = {
            deadnix.enable = true;
            nixfmt.enable = true;
          };
        };
    };
}
