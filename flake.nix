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
          runtime = self.packages.${system}.default;
          runtimePython = runtime.pythonModule.withPackages (_: [ runtime ]);
          pythonWorker = self.packages.${system}.reference-python-worker;
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
          plugin-package = self.packages.${system}.wmfs-plugin;
          python-worker-package = self.packages.${system}.reference-python-worker;

          python-worker =
            pkgs.runCommand "wmfs-python-worker-check" { nativeBuildInputs = [ pythonWorker ]; }
              ''
                env -u PYTHONPATH ${runtimePython}/bin/python3 - <<'PY'
                from pathlib import Path

                import torch

                from wmfs import add_scalar, matmul, runtime, svd
                from wmfs.plugins import find_manifests
                from wmfs.transport.worker_process import inspect_worker_environment

                plugin_directory = Path(
                    "${pythonWorker}/share/wmfs/plugins/reference"
                )
                manifest = find_manifests([plugin_directory])[0]
                environment = inspect_worker_environment(manifest)
                assert environment.python_version != "none"

                runtime.configure_control("python")
                runtime.discover_plugins(plugin_directory)
                runtime.use_backend("isolated")
                try:
                    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
                    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)
                    torch.testing.assert_close(matmul(a, b), a @ b)
                    torch.testing.assert_close(add_scalar(a, 1.5), a + 1.5)
                    differentiable_a = a.clone().requires_grad_()
                    differentiable_b = b.clone().requires_grad_()
                    expected_a = a.clone().requires_grad_()
                    expected_b = b.clone().requires_grad_()
                    loss = add_scalar(
                        matmul(differentiable_a, differentiable_b), 1.0
                    ).square().sum()
                    expected_loss = (expected_a @ expected_b + 1.0).square().sum()
                    loss.backward()
                    expected_loss.backward()
                    torch.testing.assert_close(differentiable_a.grad, expected_a.grad)
                    torch.testing.assert_close(differentiable_b.grad, expected_b.grad)
                    u, s, vh = svd(a, full_matrices=False)
                    torch.testing.assert_close(u @ torch.diag(s) @ vh, a)
                finally:
                    runtime.close()
                PY
                touch $out
              '';

          schemas = pkgs.runCommand "wmfs-schema-check" { nativeBuildInputs = [ pkgs.capnproto ]; } ''
            capnp compile -o- \
              --src-prefix=${./packages/wmfs-plugin/wmfs_plugin/schemas} \
              --import-path=${./packages/wmfs-plugin/wmfs_plugin/schemas} \
              ${./packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/runtime.capnp} >/dev/null
            capnp compile -o- \
              --src-prefix=${./packages/wmfs-plugin/wmfs_plugin/schemas} \
              --import-path=${./packages/wmfs-plugin/wmfs_plugin/schemas} \
              ${./packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/tensor.capnp} >/dev/null
            capnp compile -o- \
              --src-prefix=${./plugins/reference/schemas} \
              --import-path=${./packages/wmfs-plugin/wmfs_plugin/schemas} \
              --import-path=${./plugins/reference/schemas} \
              ${./plugins/reference/schemas/wmfs-reference/reference.capnp} >/dev/null
            touch $out
          '';

          generated =
            pkgs.runCommand "wmfs-generated-check"
              { nativeBuildInputs = [ self.packages.${system}.wmfs-plugin ]; }
              ''
                wmfs-plugin-codegen --check \
                  --schema ${./plugins/reference/schemas/wmfs-reference/reference.capnp} \
                  --python-output ${./plugins/reference/wmfs_reference/_generated.py} \
                  --cpp-output ${./plugins/reference/generated/reference_dispatch.inc}
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
              export PYTHONPATH="$development_output:$repo_root/packages/wmfs:$repo_root/packages/wmfs-plugin''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        }
      );
    };
}
