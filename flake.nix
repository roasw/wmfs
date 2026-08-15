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
      packages = forSystems (
        system:
        import ./nix/packages.nix {
          pkgs = pkgsFactory system nixpkgs;
          source = ./.;
        }
      );

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

              just-fmt = {
                enable = true;
                entry = "${pkgs.just}/bin/just --fmt --check";
                files = "(^|/)justfile$";
                pass_filenames = false;
              };

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

          package = self.packages.${system}.default;
          bundled-package = self.packages.${system}.bundled;

          schemas = pkgs.runCommand "wmfs-schema-check" { nativeBuildInputs = [ pkgs.capnproto ]; } ''
            capnp compile -o- \
              --src-prefix=${./wmfs/schemas} \
              --import-path=${./wmfs/schemas} \
              ${./wmfs/schemas/wmfs/runtime.capnp} >/dev/null
            capnp compile -o- \
              --src-prefix=${./wmfs/schemas} \
              --import-path=${./wmfs/schemas} \
              ${./wmfs/schemas/wmfs/tensor.capnp} >/dev/null
            capnp compile -o- \
              --src-prefix=${./plugins/reference/schemas} \
              --import-path=${./wmfs/schemas} \
              --import-path=${./plugins/reference/schemas} \
              ${./plugins/reference/schemas/wmfs-reference/reference.capnp} >/dev/null
            touch $out
          '';
        }
      );

      devShells = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
          pre-commit-check = self.checks.${system}.pre-commit-check;
        in
        {
          default = pkgs.mkShell {
            name = "wmfs-dev";

            inputsFrom = [
              self.packages.${system}.bundled
              self.packages.${system}.reference-worker
            ];
            packages = pre-commit-check.enabledPackages ++ [
              pkgs.just
              pkgs.python3Packages.pytest
            ];

            shellHook = pre-commit-check.shellHook + ''
              repo_root="$(git rev-parse --show-toplevel)"
              export WMFS_BUILD_TYPE="''${WMFS_BUILD_TYPE:-Debug}"
              development_output="$repo_root/output/$WMFS_BUILD_TYPE"
              export PATH="$development_output/bin:$PATH"
              export PYTHONPATH="$development_output:$repo_root''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        }
      );
    };
}
