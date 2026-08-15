{
  pkgs,
  source ? ../.,
}:
let
  pluginSrc = source + "/plugins/reference";
in
{
  reference-python-worker = pkgs.python3Packages.buildPythonApplication {
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
      substituteInPlace "$out/share/wmfs/plugins/reference/plugin.toml" \
        --replace-fail \
        'worker = "wmfs-reference-worker"' \
        'worker = "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };

  reference-worker = pkgs.stdenv.mkDerivation {
    pname = "wmfs-reference-worker";
    version = "0.1.0";
    src = source;

    nativeBuildInputs = with pkgs; [
      capnproto
      cmake
      ninja
    ];
    buildInputs = [
      pkgs.capnproto
      pkgs.python3Packages.torch.dev
      pkgs.python3Packages.torch.lib
    ];
    cmakeFlags = [
      "-DWMFS_BUILD_PYTHON_RUNTIME=OFF"
      "-DWMFS_BUILD_REFERENCE_WORKER=ON"
      "-DWMFS_BUNDLED_PLUGINS="
    ];

    postInstall = ''
      substituteInPlace "$out/share/wmfs/plugins/reference/plugin.toml" \
        --replace-fail \
        'worker = "wmfs-reference-worker"' \
        'worker = "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };
}
