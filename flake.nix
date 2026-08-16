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
      versions = import ./nix/version.nix {
        inherit self;
        lib = nixpkgs.lib;
      };
    in
    {
      packages = forSystems (
        system:
        import ./nix/packages.nix {
          pkgs = pkgsFactory system nixpkgs;
          source = ./.;
          inherit versions;
        }
      );

      checks = forSystems (
        system:
        import ./nix/checks.nix {
          gitHooks = git-hooks.lib.${system};
          packages = self.packages.${system};
          pkgs = pkgsFactory system nixpkgs;
          source = ./.;
          inherit versions;
        }
      );

      apps = forSystems (system: {
        benchmark = {
          type = "app";
          program = "${self.packages.${system}.benchmark}/bin/wmfs-benchmark";
          meta.description = "Run the packaged hermetic WMFS benchmark";
        };
      });

      devShells = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
          pre-commit-check = self.checks.${system}.pre-commit-check;
          documentationPython = pkgs.python3.withPackages (ps: [
            ps.breathe
            ps.myst-parser
            ps.numpy
            ps.pycapnp
            ps.sphinx
            ps.torch
          ]);
          withoutWmfsPackages =
            dependencies:
            builtins.filter (
              dependency:
              !(builtins.elem (pkgs.lib.getName dependency) [
                "wmfs"
                "wmfs-plugin"
                "wmfs-reference"
              ])
            ) dependencies;
          developmentRuntimeInputs = self.packages.${system}.bundled.overridePythonAttrs (previous: {
            dependencies = withoutWmfsPackages previous.dependencies;
            nativeCheckInputs = [ ];
            propagatedBuildInputs = withoutWmfsPackages (previous.propagatedBuildInputs or [ ]);
          });
          developmentWorkerInputs = self.packages.${system}.reference-worker.overrideAttrs (previous: {
            doCheck = false;
            propagatedBuildInputs = withoutWmfsPackages (previous.propagatedBuildInputs or [ ]);
          });
        in
        {
          default = pkgs.mkShell {
            name = "wmfs-dev";

            inputsFrom = [
              developmentRuntimeInputs
              developmentWorkerInputs
            ];
            packages = pre-commit-check.enabledPackages ++ [
              pkgs.doxygen
              pkgs.just
              pkgs.python3Packages.build
              pkgs.python3Packages.pytest
              pkgs.python3Packages.setuptools
              pkgs.python3Packages.setuptools-scm
              pkgs.python3Packages.wheel
              documentationPython
            ];

            shellHook = pre-commit-check.shellHook + ''
              repo_root="$(git rev-parse --show-toplevel)"
              revision="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
              dirty_suffix=""
              if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
                dirty_suffix="-dirty"
              fi
              export WMFS_GIT_VERSION="$revision$dirty_suffix"
              python_version="0.0.0+g$revision''${dirty_suffix:+.dirty}"
              export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS="$python_version"
              export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_PLUGIN="$python_version"
              export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_REFERENCE="$python_version"
              export WMFS_BUILD_TYPE="''${WMFS_BUILD_TYPE:-Debug}"
              development_output="$repo_root/output/$WMFS_BUILD_TYPE"
              export PATH="$development_output/bin:$PATH"
              export PYTHONPATH="$development_output:$repo_root/packages/wmfs:$repo_root/packages/wmfs-plugin''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        }
      );
    };
}
