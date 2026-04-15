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
        # Pip binary wheels may need extra shared libs from the store at load time. The motivating
        # case was manylinux/Linux (LD_LIBRARY_PATH + ld.so); Darwin uses dyld and DYLD_LIBRARY_PATH.
        pythonWheelRuntimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          openssl
          libffi
        ];
        libPath = pkgs.lib.makeLibraryPath pythonWheelRuntimeLibs;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.python312
          ];

          shellHook =
            ''
              export UV_PYTHON="${pkgs.python312}/bin/python3"
              export UV_PYTHON_PREFERENCE="only-system"
            ''
            + (
              if pkgs.stdenv.isLinux then
                ''
                  export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                ''
              else if pkgs.stdenv.isDarwin then
                ''
                  export DYLD_LIBRARY_PATH="${libPath}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
                ''
              else
                ""
            );
        };
      });
}
