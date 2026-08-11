{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      git-hooks,
    }:
    let
      forSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
      ];

      pkgsFactory =
        system: nixpkgs:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
    in
    {
      checks = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
        in
        {
          pre-commit-check = git-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              nixfmt.enable = true;

              gitlint.enable = true;

              mdformat = {
                package = pkgs.mdformat.withPlugins (
                  ps: with ps; [
                    mdformat-myst
                    mdformat-gfm
                  ]
                );
                enable = true;
              };

              clang-format.enable = true;

              ruff.enable = true;
              ruff-format.enable = true;
            };
          };
        }
      );

      devShells = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
          inherit (self.checks.${system}.pre-commit-check) shellHook;
        in
        {
          default = pkgs.mkShell {
            name = "wmfs-dev";

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              doxygen
              graphviz
            ];

            buildInputs = with pkgs; [
              python3
              python3Packages.torch
              python3Packages.pybind11
              python3Packages.matplotlib
            ];

            shellHook = shellHook + ''
              export PATH=$(pwd)/tools:$(pwd)/build/Debug:$PATH
              export PYTHONPATH=$(pwd)/python:$PYTHONPATH
            '';
          };
        }
      );
    };
}
