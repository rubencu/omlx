# Build and install the experimental Flash1017 fork

This fork maintains the runtime implementation. Its companion research record
belongs in `rubencu/flash1017-research`. The older prototype measurements remain
historical until this fork's exact installable artifact has been validated and
measured. A successful package build alone establishes neither inference
correctness nor the historical 20.06% first-token improvement.

The initial fork branch is `flash1017-preview`, based on public oMLX
`v0.7.1rc1` commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`. The experimental
distribution version is **`0.7.1rc1+flash1017.2`**. It is not a new model or a
new weight conversion.

Version 2 is a correctness-first candidate, **not yet quality-approved**. It
retains stock attention arithmetic, 2K prefill and native acceptance-based MTP
(maximum depth three), plus head reuse for rows two through five, Engram row
caching and 64 MiB prefetch. The older replay/confidence/rounding profile and
its performance results do not describe this version. Use a fresh server base
directory; the new `flash1017-v2` prefix-cache namespace does not reuse v1 state.

## What the wheel contains

The helper builds **all Python modules, web files and declared package data
from the current fork checkout**. It reuses only five native extension modules
and their five Metal libraries from the [public oMLX v0.7.1rc1 release](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1).
No installed oMLX application is read or copied. The wheel includes the original
native license, fork notices, package licenses, regenerated metadata/RECORD and
a `FLASH1017-BUILD.json` provenance record in its distribution metadata.

This route is useful without a full Xcode installation. It is a source-derived
experimental fork with reused public native assets. It is not a rebuild of the
native code and is not byte-identical to the historical app's native binaries.
The ordinary upstream source/native build remains available unchanged.

This artifact installs the oMLX server and web interface. It does not replace
the existing macOS menu-bar application or ship a rebuilt app bundle. After
launch, open `http://127.0.0.1:8017/admin` (or the port you chose) and follow the
ordinary initial setup. The model loads on the first request.

The base filename says `universal2`, but all five pinned native modules are
thin arm64 Mach-O bundles. Our wheel therefore uses the narrower, accurate tag
**`cp312-cp312-macosx_15_0_arm64`**: CPython 3.12, Apple Silicon and a macOS 15
minimum binary deployment target. This does not qualify the optimized model on
every compatible Mac or establish Intel support.

## Build from the checkout

Requirements are Git, CPython 3.12 and the pinned build tools below. The helper
itself does not download dependencies, import MLX, install anything or compile
native kernels. It uses the existing setuptools build backend in a clean
temporary staging tree. Building the artifact does not require model weights.

```sh
python3.12 -m venv .venv-flash1017-build
.venv-flash1017-build/bin/python -m pip install setuptools==84.0.0 wheel==0.45.1
mkdir -p downloads
curl --fail --location \
  https://github.com/jundot/omlx/releases/download/v0.7.1rc1/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl \
  --output downloads/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl
.venv-flash1017-build/bin/python -I -B scripts/build_flash1017_wheel.py \
  --base-wheel downloads/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl \
  --output-dir dist/flash1017
```

The base wheel must have SHA-256:

```text
b3e12950a607991a1feb833b1bdc0b92fe41aeb73656ca5373db03d0d87b71c7
```

The Git checkout must contain the pinned base commit. The helper compares the
entire native package source tree, including its Python wrappers, and
`setup.py` to that commit. It refuses changed, missing or additional native
source. It also requires the runtime's exact `mlx==0.32.4` pin and the fork
version. Existing local compiled files and bytecode cannot enter the build.
Changing native code or the MLX ABI requires a real native rebuild and a new
packaging contract; there is no override to bypass these checks.

The output is:

```text
omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl
omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl.sha256
```

The helper refuses an existing wheel filename. To compare repeated builds,
use separate output directories. File order, timestamps and modes are fixed;
source hashes and tool versions are recorded without private workspace paths.
Byte reproducibility still assumes the same source tree, helper, Python/build
tools and compression implementation. This is not a fully locked runtime
dependency environment: installation resolves the remaining dependencies
declared in `pyproject.toml`, including its exact Git revisions.

## Install in a separate environment

Use a separate environment and data directory to keep the experimental fork's
dependencies and server settings distinct from another oMLX installation.
Installation needs network access unless all runtime dependencies are cached.

```sh
python3.12 -m venv .venv-flash1017
.venv-flash1017/bin/python -m pip install \
  dist/flash1017/omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl
.venv-flash1017/bin/python -m pip check
.venv-flash1017/bin/omlx --version
```

The Flash1017 entry point targets the unchanged
[`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp` checkpoint at revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab).
Use the local model directory for that snapshot. The command checks pinned
metadata before allocation; metadata validation does not rehash all weight
shards. The wheel contains no model weights.

```sh
.venv-flash1017/bin/omlx flash1017 --model /path/to/the/pinned/model --check
.venv-flash1017/bin/omlx flash1017 --model /path/to/the/pinned/model \
  --base-path ~/.omlx-flash1017 --port 8017
```

`--check` performs the non-loading checks. `--configure-only` prepares the
isolated profile without starting the server. The default experimental base
directory is `~/.omlx-flash1017`; the ordinary server defaults are not the
experimental profile. Choose an unused port. Stop a foreground server with
Ctrl-C.

The historical optimized profile was tested on a 256 GiB M5 Ultra, with one
active request, queued requests and a 131,072-token **total** context cap.
The timing suite reached 64K input tokens. Longer offline checks do not qualify
million-token chat, and the separate long-serving gate failed. Those limits
remain visible in the research repository; the new fork needs its own recorded
installation, serving and numerical checks before any performance claim.

## Packaging checks without inference

```sh
.venv-flash1017-build/bin/python -I -B tests/test_flash1017_packaging.py -v
```

Run this test file directly: the broad upstream pytest configuration imports
GPU/runtime fixtures. These tests instead use synthetic archives and source
fixtures to exercise rejected hashes, native-source changes, missing assets,
stale Python files, accurate tags/licenses, complete RECORD hashes and
deterministic ZIP output. They do not load the model or qualify its answers.

The helper uses the existing [setuptools build backend](https://setuptools.pypa.io/en/latest/build_meta.html)
and emits the metadata and RECORD required by the
[wheel specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
Publication and upstream submissions remain separate actions requiring review.
