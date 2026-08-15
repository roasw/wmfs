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
      pluginSrc = wmfs.outPath + "/plugins/reference";

      reference-worker = pkgs.python3Packages.buildPythonApplication {
        pname = "wmfs-reference";
        version = "0.1.0";
        pyproject = true;
        src = pluginSrc;

        build-system = [ pkgs.python3Packages.setuptools ];
        dependencies = with pkgs.python3Packages; [
          numpy
          pycapnp
          torch
        ];

        pythonImportsCheck = [ "wmfs_reference" ];

        postInstall = ''
          install -Dm444 \
            ${pluginSrc}/plugin.toml \
            "$out/share/wmfs/plugins/reference/plugin.toml"
          substituteInPlace "$out/share/wmfs/plugins/reference/plugin.toml" \
            --replace-fail \
            'worker = "wmfs-reference-worker"' \
            'worker = "'"$out"'/bin/wmfs-reference-worker"'
          install -Dm444 \
            ${pluginSrc}/schemas/wmfs-reference/reference.capnp \
            "$out/share/wmfs/plugins/reference/schemas/wmfs-reference/reference.capnp"
        '';
      };

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

            from wmfs import add_scalar, runtime
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
            assert worker.python_version == "${pkgs.python3.version}"
            assert worker.torch_version.startswith("${pkgs.python3Packages.torch.version}")
            assert worker.executable.startswith("${pkgs.python3}")

            runtime.discover_plugins(plugin_directory)
            runtime.use_backend("isolated")
            try:
                source = torch.arange(4, dtype=torch.float64)
                result = add_scalar(source, 1.5)
                torch.testing.assert_close(result, source + 1.5)
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
                "${pkgs.python3}",
                "${pkgs.python3Packages.pycapnp}",
                "${pkgs.python3Packages.torch}",
            )
            assert not any(path in process_maps for path in worker_closure_paths)
            PY
            touch "$out"
          '';
    in
    {
      packages.${system} = {
        default = reference-worker;
        inherit reference-worker;
      };

      apps.${system} = {
        default = self.apps.${system}.reference-worker;
        reference-worker = {
          type = "app";
          program = "${reference-worker}/bin/wmfs-reference-worker";
          meta.description = "Run the glibc 2.40 reference worker";
        };
      };

      checks.${system} = {
        default = isolation-check;
        package = reference-worker;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.capnproto
          reference-worker
          (pkgs.python3.withPackages (
            ps: with ps; [
              numpy
              pycapnp
              pytest
              torch
            ]
          ))
        ];
      };
    };
}
