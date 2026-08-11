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
        let
          pkgs = pkgsFactory system nixpkgs;
          reference-worker = pkgs.python3Packages.buildPythonApplication {
            pname = "wmfs-reference";
            version = "0.1.0";
            pyproject = true;
            src = ./plugins/reference;

            build-system = [ pkgs.python3Packages.setuptools ];
            dependencies = with pkgs.python3Packages; [
              pycapnp
              torch
            ];

            pythonImportsCheck = [ "wmfs_reference" ];
          };
        in
        {
          default = pkgs.python3Packages.buildPythonPackage {
            pname = "wmfs";
            version = "0.1.0";
            pyproject = true;
            src = ./.;

            build-system = [ pkgs.python3Packages.setuptools ];
            dependencies = with pkgs.python3Packages; [
              pycapnp
              torch
            ];

            nativeCheckInputs = [
              pkgs.python3Packages.pytestCheckHook
              reference-worker
            ];
            pythonImportsCheck = [ "wmfs" ];
          };

          inherit reference-worker;
        }
      );

      apps = forSystems (system: {
        reference-worker = {
          type = "app";
          program = "${self.packages.${system}.reference-worker}/bin/wmfs-reference-worker";
          meta.description = "Run the wmfs reference plugin worker";
        };
      });

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

          package = self.packages.${system}.default;

          schemas = pkgs.runCommand "wmfs-schema-check" { nativeBuildInputs = [ pkgs.capnproto ]; } ''
            capnp compile -o- \
              --src-prefix=${./wmfs/schemas} \
              --import-path=${./wmfs/schemas} \
              ${./wmfs/schemas/wmfs/runtime.capnp} >/dev/null
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
          python = pkgs.python3.withPackages (
            ps: with ps; [
              matplotlib
              nanobind
              pycapnp
              pytest
              torch
            ]
          );
        in
        {
          default = pkgs.mkShell {
            name = "wmfs-dev";

            nativeBuildInputs =
              pre-commit-check.enabledPackages
              ++ (with pkgs; [
                capnproto
                cmake
                ninja
                doxygen
                graphviz
              ]);

            buildInputs = [
              python
              self.packages.${system}.reference-worker
            ];

            shellHook = pre-commit-check.shellHook + ''
              repo_root="$(git rev-parse --show-toplevel)"
              export PATH="$repo_root/tools:$repo_root/build/Debug:$PATH"
              export PYTHONPATH="$repo_root''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        }
      );
    };
}
