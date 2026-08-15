{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    wmfs.url = "path:../..";
    nixpkgs-current.follows = "wmfs/nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-current,
      wmfs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      source = wmfs.outPath;
      workers = import (source + "/nix/reference-workers.nix") {
        inherit pkgs source;
      };
      inherit (workers) reference-python-worker reference-worker;

      runtime = wmfs.packages.${system}.default;
      runtimePython = runtime.pythonModule.withPackages (_: [ runtime ]);

      isolation-check =
        pkgs.runCommand "wmfs-nixos-25.05-isolation"
          {
            nativeBuildInputs = [
              reference-worker
            ];
          }
          ''
            env -u PYTHONPATH ${runtimePython}/bin/python3 - <<'PY'
            import ctypes
            import sys
            from pathlib import Path

            import torch

            from wmfs import add_scalar, matmul, runtime, svd
            from wmfs.plugins import find_manifests
            from wmfs.transport.worker_process import inspect_worker_environment

            plugin_directory = Path(
                "${reference-worker}/share/wmfs/plugins/reference"
            )
            manifest = find_manifests([plugin_directory])[0]
            assert manifest.worker == "${reference-worker}/bin/wmfs-reference-worker"
            worker = inspect_worker_environment(manifest)

            libc = ctypes.CDLL(None)
            libc.gnu_get_libc_version.restype = ctypes.c_char_p
            runtime_glibc = libc.gnu_get_libc_version().decode()

            assert runtime_glibc == "${nixpkgs-current.legacyPackages.${system}.glibc.version}"
            assert worker.glibc_version == "${pkgs.glibc.version}"
            assert worker.glibc_version == "2.40"
            assert worker.glibc_version != runtime_glibc
            assert worker.python_version == "none"
            assert worker.torch_version == "${pkgs.python3Packages.torch.version}"
            assert worker.executable == "${reference-worker}/bin/wmfs-reference-worker"

            runtime.discover_plugins(plugin_directory)
            runtime.use_backend("isolated")
            try:
                source = torch.arange(4, dtype=torch.float64)
                result = add_scalar(source, 1.5)
                torch.testing.assert_close(result, source + 1.5)

                matrix = torch.arange(12, dtype=torch.float64).reshape(4, 3)
                product = matmul(matrix, matrix.T)
                torch.testing.assert_close(product, matrix @ matrix.T)

                u, singular_values, vh = svd(matrix, full_matrices=False)
                torch.testing.assert_close(
                    u @ torch.diag(singular_values) @ vh,
                    matrix,
                )
            finally:
                runtime.close()

            assert not any(
                name == "wmfs_reference" or name.startswith("wmfs_reference.")
                for name in sys.modules
            )
            process_maps = Path("/proc/self/maps").read_text()
            worker_closure_paths = (
                "${reference-worker}",
                "${pkgs.glibc}",
                "${pkgs.python3Packages.torch.lib}",
            )
            assert not any(path in process_maps for path in worker_closure_paths)
            PY
            touch "$out"
          '';
    in
    {
      packages.${system} = {
        default = reference-worker;
        inherit reference-python-worker reference-worker;
      };

      checks.${system} = {
        default = isolation-check;
        package = reference-worker;
      };

      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [
          reference-python-worker
          reference-worker
        ];
        packages = [
          pkgs.python3Packages.pytest
          reference-worker
        ];
      };
    };
}
