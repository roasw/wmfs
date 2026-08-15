{
  pkgs,
  source ? ../.,
}:
let
  pluginSrc = source + "/plugins/reference";
  wmfsPlugin = import ./wmfs-plugin.nix { inherit pkgs source; };
in
{
  wmfs-plugin = wmfsPlugin;

  reference-python-worker = pkgs.python3Packages.buildPythonApplication {
    pname = "wmfs-reference";
    version = "0.1.0";
    pyproject = true;
    src = pluginSrc;

    build-system = [ pkgs.python3Packages.setuptools ];
    dependencies = [
      pkgs.python3Packages.torch
      wmfsPlugin
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
