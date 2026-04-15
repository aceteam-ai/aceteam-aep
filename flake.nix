{
  description = "aceteam-aep development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Shared libs manylinux/pip wheels expect at load time. Extend this list as needed.
        pythonWheelRuntimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.python312
          ];

          # Makes the above store paths visible to ld.so when Python loads extension modules.
          shellHook = ''
            export UV_PYTHON="${pkgs.python312}/bin/python3"
            export UV_PYTHON_PREFERENCE="only-system"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath pythonWheelRuntimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });
}
