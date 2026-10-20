# Build and run DeepSeek Flash1017

This guide builds `0.7.1rc1+flash1017.2` from the `flash1017-preview` checkout
and runs an isolated oMLX server with the Flash1017 profile enabled. **No fork
release has been published yet.** The upstream app, Homebrew formula and
release downloads install upstream oMLX; use the wheel built here for this
profile.

The profile combines BF16 vocabulary-weight reuse for two through five
verification rows, a bounded Engram row cache and 64 MiB Engram prefetch. It
retains stock attention arithmetic, prompt chunks of up to 2,048 tokens and
native acceptance-based MTP capped at three drafts. The
[profile guide](flash1017.md) covers implementation, results and limits.

## Requirements

- **Apple Silicon, macOS 15 or later, CPython 3.12.** The evaluated machine is
  a 256 GiB M5 Ultra Mac Studio. Other Macs have not been evaluated.
- **At least 256 GiB unified memory and 210 GiB available before model loading.**
  Unload other models first. The loader also checks recommended working-set
  capacity and maintains a 12 GiB available-memory floor around shard loading.
- **The complete pinned checkpoint**, stored locally:
  [`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
  revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. The wheel contains no
  model weights and the server does not convert or copy them.
- **Git and this repository checkout**, including public base commit
  `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`, for building. Run the commands
  below from the repository root. Building alone does not need model weights
  or a 256 GiB machine.

The server allows one active request, queues further requests and caps total
context at **131,072 tokens including output**. Its process-wide MLX memory
settings reserve 32 GiB of physical capacity in the limit calculation and set
an 8 GiB allocator-cache guideline. These are not total-process RSS limits or
a guarantee that every request fits.

## Build the wheel

The helper packages this checkout's Python modules, web interface and declared
data files. It reuses five native extensions and their five Metal libraries
from the public oMLX `v0.7.1rc1` wheel, so this build route does not require full
Xcode or compile native kernels. It does not read an installed application.

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

The helper verifies the base wheel against this SHA-256:

```text
b3e12950a607991a1feb833b1bdc0b92fe41aeb73656ca5373db03d0d87b71c7
```

It also compares the native package source tree, including Python wrappers,
and `setup.py` with the pinned public base. Missing, added or changed native
sources are rejected. The exact `mlx==0.32.4` pin is required. Local compiled
files and bytecode cannot enter the build; changes to native code or the MLX
ABI require separately compiled and validated assets.

The output files are:

```text
omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl
omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl.sha256
```

The base filename says `universal2`, but its five native modules are thin arm64
Mach-O bundles. The resulting wheel therefore uses the accurate
`cp312-cp312-macosx_15_0_arm64` tag. It does not support Intel Macs.

The helper refuses to overwrite an existing wheel; use a different output
directory for another build. It fixes file ordering, timestamps and modes and
records source hashes and tool versions. Byte reproducibility also depends on
the same source tree, Python, build tools and compression implementation.

## Install in a separate environment

Keep this server's dependencies separate from another oMLX installation.
Installation needs network access unless all dependencies are cached.

```sh
python3.12 -m venv .venv-flash1017
.venv-flash1017/bin/python -m pip install \
  dist/flash1017/omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl
.venv-flash1017/bin/python -m pip check
.venv-flash1017/bin/omlx --version
```

Expect `0.7.1rc1+flash1017.2`. Installation resolves the runtime dependencies
in `pyproject.toml`, including its exact MLX and Git revision pins; it is not
a fully locked dependency environment. Reproducing the reported measurements
also requires the dependency versions recorded with their evidence.

## Validate and launch

Replace `/path/to/the/pinned/model` with the directory containing the complete
checkpoint snapshot, including its metadata and weight shards.

```sh
.venv-flash1017/bin/omlx flash1017 \
  --model /path/to/the/pinned/model --check
.venv-flash1017/bin/omlx flash1017 \
  --model /path/to/the/pinned/model \
  --base-path ~/.omlx-flash1017 --port 8017
```

`--check` verifies four pinned metadata hashes without loading weights. It
does not hash every weight shard or prove that the machine has sufficient
loading headroom. To prepare settings without starting the server, use
`--configure-only` instead.

Choose an empty base directory or one already configured for this profile and
checkpoint. The command creates a symlink to the model and enables the
model-local `deepseek_v41_flash1017_enabled` setting; ordinary oMLX defaults
leave it disabled. The default base directory is `~/.omlx-flash1017`, and the
prefix-cache namespace is `flash1017-v2`. Settings and caches stay separate
from other installations. The command refuses an unrelated nonempty base
directory.

The example binds to loopback on port **8017**; choose an unused port. If
`--port` is omitted, the CLI defaults to **8001**. Open
`http://127.0.0.1:8017/admin` and complete the normal initial setup. The model
loads on the first request. OpenAI-compatible clients use
`http://127.0.0.1:8017/v1` with model ID `DeepSeek-V4.1-Flash-1017` and the
server's configured authentication.

A fresh configuration disables thinking and uses temperature 1.0. The reported
quality/performance comparisons instead supplied greedy temperature 0 in each
request. Set request sampling explicitly when reproducing those results.
The launch workflow configures a 4 GB hot prefix cache and 20 GB SSD cache
budget; the performance comparison disabled prefix reuse. These caches are
separate from the Engram row cache, which retains up to 512 MiB.

Stop this foreground server with **Ctrl-C** and wait for it to exit before
launching another large-model process or changing its configuration. Normal
shutdown drains the inference worker and prefetch reader before releasing
resources. The wheel installs the server and web interface; it does not
replace the macOS menu-bar app or install a background service.

## Tested artifact and evidence

The evaluated runtime commit is `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`.
The preserved wheel's SHA-256 is:

```text
0fbdb77ca86ce35b4dca0275b534c1fafde7d70f0f7f6a89d5f964821f7f7f06
```

That wheel matched all **618 tested baseline answers and finish reasons**
exactly. Its complete eight-workload comparison measured **7.9% higher
generation throughput and 4.5% lower first-visible-token latency** on one
256 GiB M5 Ultra, with 32 measured requests in two counterbalanced blocks.
See the [methods and limits](flash1017.md#tested-results) before applying
those figures to a different setup.

A build from this checkout may have a different checksum because documentation
and build metadata are included. Do not use the tested checksum as the expected
checksum for every rebuild. The companion `flash1017-research` checkout holds
the exact wheel/dependency identities and evaluation evidence; it has not yet
been published.

## Native provenance and packaging checks

The wheel preserves package licenses, the upstream native license and fork
notices. Its distribution metadata contains `FLASH1017-BUILD.json` with source,
base-wheel and native-asset identities, plus a regenerated wheel `RECORD`.
The build reuses public native assets; it is not a native-source rebuild.
The upstream source/native build route remains available.

Packaging tests can run without inference:

```sh
.venv-flash1017-build/bin/python -I -B tests/test_flash1017_packaging.py -v
```

Run this file directly because the broad upstream pytest setup imports runtime
fixtures. These tests use synthetic archives to check rejected hashes, native
source changes, missing assets, package contents, tags, licenses, `RECORD`
hashes and deterministic ZIP output. They do not load a model or validate its
answers.
