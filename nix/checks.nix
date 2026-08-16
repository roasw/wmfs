{
  gitHooks,
  packages,
  pkgs,
  source,
  versions,
}:
let
  runtime = packages.default;
  runtimePython = runtime.pythonModule.withPackages (_: [ runtime ]);
  benchmark = packages.benchmark;
  bundledRuntime = packages.bundled;
  referenceWorker = packages.reference-worker;
  pythonWorker = packages.reference-python-worker;
  documentationPython = pkgs.python3.withPackages (ps: [
    ps.breathe
    ps.myst-parser
    ps.numpy
    ps.pycapnp
    ps.sphinx
    ps.torch
  ]);
in
{
  pre-commit-check = gitHooks.run {
    src = source;
    hooks = {
      nixfmt.enable = true;

      actionlint = {
        enable = true;
        entry = "${pkgs.actionlint}/bin/actionlint";
        files = "^\\.github/workflows/.*\\.ya?ml$";
      };

      yamlfmt = {
        enable = true;
        entry = "${pkgs.yamlfmt}/bin/yamlfmt";
        files = "\\.ya?ml$";
      };

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

  package = packages.default;
  bundled-package = packages.bundled-check;
  benchmark-package = benchmark;
  plugin-package = packages.wmfs-plugin;
  python-worker-package = packages.reference-python-worker;
  python-artifacts = import ./artifacts-check.nix {
    inherit pkgs source versions;
  };

  python-worker =
    pkgs.runCommand "wmfs-python-worker-check" { nativeBuildInputs = [ pythonWorker ]; }
      ''
        env -u PYTHONPATH ${runtimePython}/bin/python3 - <<'PY'
        from pathlib import Path

        import torch
        import wmfs

        from wmfs.plugins import find_manifests
        from wmfs.transport.worker_process import inspect_worker_environment

        runtime = wmfs.runtime

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
            torch.testing.assert_close(wmfs.matmul(a, b), a @ b)
            torch.testing.assert_close(wmfs.add_scalar(a, 1.5), a + 1.5)
            differentiable_a = a.clone().requires_grad_()
            differentiable_b = b.clone().requires_grad_()
            expected_a = a.clone().requires_grad_()
            expected_b = b.clone().requires_grad_()
            loss = wmfs.add_scalar(
                wmfs.matmul(differentiable_a, differentiable_b), 1.0
            ).square().sum()
            expected_loss = (expected_a @ expected_b + 1.0).square().sum()
            loss.backward()
            expected_loss.backward()
            torch.testing.assert_close(differentiable_a.grad, expected_a.grad)
            torch.testing.assert_close(differentiable_b.grad, expected_b.grad)
            u, s, vh = wmfs.svd(a, full_matrices=False)
            torch.testing.assert_close(u @ torch.diag(s) @ vh, a)
        finally:
            runtime.close()
        PY
        touch $out
      '';

  benchmark-smoke =
    pkgs.runCommand "wmfs-benchmark-smoke-check" { nativeBuildInputs = [ benchmark ]; }
      ''
        mkdir -p "$TMPDIR/work" "$TMPDIR/hostile/wmfs"
        printf 'raise RuntimeError("hostile wmfs import")\n' \
          > "$TMPDIR/hostile/wmfs/__init__.py"
        cd "$TMPDIR/work"
        PYTHONPATH="$TMPDIR/hostile" \
        PYTHONHOME="$TMPDIR/hostile" \
        LD_LIBRARY_PATH="$TMPDIR/hostile" \
        LD_PRELOAD="$TMPDIR/hostile/libhostile.so" \
          wmfs-benchmark \
            --operations add_scalar \
            --tiers small \
            --add-scalar-sizes 1 1 1 \
            --iterations 1 \
            --warmups 0 \
            --startup-iterations 1 \
            --rpc-iterations 1 \
            --diagnostic-iterations 1 \
            --high-frequency-iterations 1 \
            --format json \
            --output report.json
        env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
          ${pkgs.python3}/bin/python3 - <<'PY'
        import json
        from pathlib import Path

        report = json.loads(Path("report.json").read_text())
        runtime_module = Path(report["environment"]["wmfs_module"])
        plugin_directory = Path(report["configuration"]["plugin_directory"])
        worker = Path(report["environment"]["worker"]["executable"])

        assert runtime_module.is_relative_to(Path("${bundledRuntime}"))
        assert plugin_directory == Path(
            "${referenceWorker}/share/wmfs/plugins/reference"
        )
        assert worker == Path(
            "${referenceWorker}/bin/wmfs-reference-worker"
        )
        assert report["operations"][0]["operation"] == "add_scalar"
        PY
        touch $out
      '';

  schemas = pkgs.runCommand "wmfs-schema-check" { nativeBuildInputs = [ pkgs.capnproto ]; } ''
    capnp compile -o- \
      --src-prefix=${source}/packages/wmfs-plugin/wmfs_plugin/schemas \
      --import-path=${source}/packages/wmfs-plugin/wmfs_plugin/schemas \
      ${source}/packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/runtime.capnp >/dev/null
    capnp compile -o- \
      --src-prefix=${source}/packages/wmfs-plugin/wmfs_plugin/schemas \
      --import-path=${source}/packages/wmfs-plugin/wmfs_plugin/schemas \
      ${source}/packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/tensor.capnp >/dev/null
    capnp compile -o- \
      --src-prefix=${source}/plugins/reference/schemas \
      --import-path=${source}/packages/wmfs-plugin/wmfs_plugin/schemas \
      --import-path=${source}/plugins/reference/schemas \
      ${source}/plugins/reference/schemas/wmfs-reference/reference.capnp >/dev/null
    touch $out
  '';

  generated =
    pkgs.runCommand "wmfs-generated-check" { nativeBuildInputs = [ packages.wmfs-plugin ]; }
      ''
        wmfs-plugin-codegen --check \
          --schema ${source}/plugins/reference/schemas/wmfs-reference/reference.capnp \
          --python-output ${source}/plugins/reference/wmfs_reference/_generated.py \
          --cpp-output ${source}/plugins/reference/generated/reference_dispatch.inc
        touch $out
      '';

  documentation = pkgs.stdenv.mkDerivation {
    name = "wmfs-documentation";
    src = source;
    WMFS_GIT_VERSION = versions.git;
    nativeBuildInputs = [
      pkgs.capnproto
      pkgs.cmake
      pkgs.doxygen
      pkgs.ninja
      documentationPython
    ];
    buildInputs = [ pkgs.capnproto ];
    cmakeFlags = [
      "-DBUILD_TESTING=OFF"
      "-DWMFS_VERSION=${versions.git}"
      "-DWMFS_BUILD_DOCUMENTATION=ON"
      "-DWMFS_BUILD_PYTHON_RUNTIME=OFF"
      "-DWMFS_BUILD_REFERENCE_WORKER=OFF"
    ];
    buildPhase = ''
      runHook preBuild
      cmake --build . --target doc
      runHook postBuild
    '';
    installPhase = ''
      mkdir -p "$out"
      cp -R "$NIX_BUILD_TOP/$sourceRoot/build/docs/html/." "$out/"
    '';
  };
}
