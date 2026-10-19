#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build this fork's sources with hash-pinned public native assets, offline.

No oMLX/MLX imports, application-bundle reads, downloads or native compilation.
The ordinary upstream build remains unchanged. See docs/flash1017-install.md.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import tomllib
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BASE_COMMIT = "7a5a3d75ce7463300bdfa5fedccd34348f60ff46"
BASE_VERSION = "0.7.1rc1"
VERSION = "0.7.1rc1+flash1017.2"
BASE_WHEEL_NAME = "omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl"
BASE_WHEEL_SHA256 = "b3e12950a607991a1feb833b1bdc0b92fe41aeb73656ca5373db03d0d87b71c7"
BASE_URL = f"https://github.com/jundot/omlx/releases/download/v{BASE_VERSION}/{BASE_WHEEL_NAME}"
BASE_TAG = "cp312-cp312-macosx_15_0_universal2"
# The five binaries in the pinned upstream wheel are arm64, not universal2.
TAG = "cp312-cp312-macosx_15_0_arm64"
DIST_INFO = f"omlx-{VERSION}.dist-info"
BASE_DIST_INFO = f"omlx-{BASE_VERSION}.dist-info"
NATIVE_ROOT = "omlx/custom_kernels/"
FLASH_ROOT = "omlx/patches/deepseek_v41/flash1017/"
PACKAGE_ROOTS = {"omlx": "omlx", "omlx_web": "apps/omlx-web/omlx_web"}
LIBRARIES = {
    "bonsai": "omlx_bonsai_kernels",
    "decode_fast": "omlx_decode_fast_kernels",
    "glm_moe_dsa": "omlx_glm_kernels",
    "minimax_m3": "omlx_minimax_m3_kernels",
    "qwen35_prefill": "omlx_qwen35_prefill_kernels",
}
NATIVE_NAMES = frozenset(
    f"{NATIVE_ROOT}{package}/{filename}"
    for package, library in LIBRARIES.items()
    for filename in ("_ext.cpython-312-darwin.so", f"{library}.metallib")
)
BUILD_TOOLS = {"setuptools": "84.0.0", "wheel": "0.45.1"}
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class BuildError(RuntimeError):
    """An input or build result violated this narrow packaging contract."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"Expected a regular source file: {path.name}")
    return path.read_bytes()


def ignored(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    )


def regular_files(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise BuildError(f"Expected a source directory, not a symlink: {root.name}")
    result = {}
    for path in sorted(root.rglob("*")):
        if ignored(path.relative_to(root)):
            continue
        if path.is_symlink():
            raise BuildError(
                f"Source symlink is not supported: {path.relative_to(root)}"
            )
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def read_zip(path: Path) -> dict[str, bytes]:
    entries = {}
    with ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            pure = PurePosixPath(name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "\\" in name
                or name != pure.as_posix()
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise BuildError(f"Unsafe wheel member: {name}")
            if info.is_dir() or name in entries:
                raise BuildError(f"Duplicate or directory wheel member: {name}")
            entries[name] = archive.read(info)
    return entries


def parse_metadata(data: bytes):
    return BytesParser().parsebytes(data, headersonly=True)


def check_arm64_bundle(data: bytes, name: str) -> None:
    if len(data) < 32:
        raise BuildError(f"Truncated native library: {name}")
    magic, cpu, _, filetype, commands, commands_size, _, _ = struct.unpack_from(
        "<8I", data
    )
    if (magic, cpu, filetype) != (0xFEEDFACF, 0x0100000C, 8):
        raise BuildError(f"Expected a thin arm64 Mach-O bundle: {name}")
    cursor = 32
    if commands_size > len(data) - cursor:
        raise BuildError(f"Invalid native load commands: {name}")
    minimums = []
    for _ in range(commands):
        if cursor + 8 > 32 + commands_size:
            raise BuildError(f"Invalid native load command: {name}")
        command, size = struct.unpack_from("<2I", data, cursor)
        if size < 8 or cursor + size > 32 + commands_size:
            raise BuildError(f"Invalid native load command size: {name}")
        if command == 0x32:  # LC_BUILD_VERSION: platform, minos, sdk, ntools
            if size < 24:
                raise BuildError(f"Invalid native build version: {name}")
            platform, minimum = struct.unpack_from("<2I", data, cursor + 8)
            if platform != 1:
                raise BuildError(f"Native library is not built for macOS: {name}")
            minimums.append(minimum)
        cursor += size
    if cursor != 32 + commands_size or not minimums or max(minimums) > (15 << 16):
        raise BuildError(f"Native deployment target exceeds the macOS 15 tag: {name}")


def load_base_assets(path: Path) -> tuple[dict[str, bytes], bytes]:
    if digest(checked_bytes(path)) != BASE_WHEEL_SHA256:
        raise BuildError("Base wheel SHA-256 differs from the pinned public release")
    entries = read_zip(path)
    metadata = parse_metadata(entries[f"{BASE_DIST_INFO}/METADATA"])
    wheel = parse_metadata(entries[f"{BASE_DIST_INFO}/WHEEL"])
    if (metadata.get("Name"), metadata.get("Version")) != ("omlx", BASE_VERSION):
        raise BuildError("Unexpected base distribution identity")
    if wheel.get_all("Tag") != [BASE_TAG] or wheel.get("Root-Is-Purelib") != "false":
        raise BuildError("Unexpected base wheel tags")
    found = {name for name in entries if name.endswith((".so", ".metallib", ".dylib"))}
    if found != NATIVE_NAMES:
        raise BuildError(
            "Base wheel native payload does not match the exact ten-file allowlist"
        )
    for name in NATIVE_NAMES:
        if name.endswith(".so"):
            check_arm64_bundle(entries[name], name)
    return {name: entries[name] for name in sorted(NATIVE_NAMES)}, entries[
        f"{BASE_DIST_INFO}/licenses/LICENSE"
    ]


def check_native_snapshot(root: Path, baseline: dict[str, bytes]) -> dict[str, str]:
    current = {
        NATIVE_ROOT + name: data
        for name, data in regular_files(root / NATIVE_ROOT).items()
        if NATIVE_ROOT + name not in NATIVE_NAMES
    }
    current["setup.py"] = checked_bytes(root / "setup.py")
    if set(current) != set(baseline):
        changes = sorted(set(current) ^ set(baseline))
        raise BuildError(
            f"Native source file set differs from the public base: {changes}"
        )
    changed = [name for name in baseline if current[name] != baseline[name]]
    if changed:
        raise BuildError(
            f"Native source differs from the public base: {sorted(changed)}"
        )
    return {name: digest(data) for name, data in sorted(current.items())}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE
    )


def verify_native_sources(root: Path) -> dict[str, str]:
    paths = git(
        root,
        "ls-tree",
        "-rz",
        "--name-only",
        BASE_COMMIT,
        "--",
        NATIVE_ROOT,
        "setup.py",
    )
    names = [name.decode("utf-8") for name in paths.split(b"\0") if name]
    if "setup.py" not in names or not any(
        name.startswith(NATIVE_ROOT) for name in names
    ):
        raise BuildError("The pinned public base is missing its native source tree")
    baseline = {name: git(root, "show", f"{BASE_COMMIT}:{name}") for name in names}
    return check_native_snapshot(root, baseline)


def collect_sources(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Select Python modules and pyproject-declared data, never local binaries."""
    config = tomllib.loads(checked_bytes(root / "pyproject.toml").decode())
    if config["build-system"]["build-backend"] != "setuptools.build_meta":
        raise BuildError("This helper requires the existing setuptools backend")
    if "mlx==0.32.4" not in config["project"]["dependencies"]:
        raise BuildError(
            "Reused native assets require the exact mlx==0.32.4 dependency"
        )
    version_tree = ast.parse(checked_bytes(root / "omlx/_version.py"))
    versions = [
        ast.literal_eval(node.value)
        for node in version_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if versions != [VERSION]:
        raise BuildError(
            f"Set omlx/_version.py to the fork version {VERSION} before building"
        )
    sources = {}
    package_paths = {}
    all_package_files = {}
    for name, relative in PACKAGE_ROOTS.items():
        files = regular_files(root / relative)
        all_package_files.update(
            {f"{relative}/{path}": data for path, data in files.items()}
        )
        for path, data in files.items():
            if path.endswith(".py"):
                source = f"{relative}/{path}"
                sources[source] = data
                package_paths[f"{name}/{path}"] = source
    package_data = config.get("tool", {}).get("setuptools", {}).get("package-data", {})
    for package, patterns in package_data.items():
        top, *rest = package.split(".")
        if top not in PACKAGE_ROOTS:
            raise BuildError(f"Unsupported package-data root: {package}")
        relative = str(PurePosixPath(PACKAGE_ROOTS[top], *rest))
        for pattern in patterns:
            if (
                PurePosixPath(pattern).is_absolute()
                or ".." in PurePosixPath(pattern).parts
            ):
                raise BuildError(f"Package-data pattern escapes its package: {pattern}")
            for path in (root / relative).glob(pattern):
                if path.is_file() and not ignored(path):
                    source = path.relative_to(root).as_posix()
                    target = str(
                        PurePosixPath(top, *rest, path.relative_to(root / relative))
                    )
                    if target in NATIVE_NAMES:
                        continue
                    if path.suffix in {".so", ".metallib", ".dylib"}:
                        raise BuildError(
                            f"Unpinned binary declared as package data: {source}"
                        )
                    sources[source] = checked_bytes(path)
                    package_paths[target] = source
    missing = sorted(
        name
        for name in all_package_files
        if name.startswith(FLASH_ROOT) and name not in sources
    )
    if missing:
        raise BuildError(
            f"Flash1017 assets must be declared in package-data: {missing}"
        )
    if not any(name.startswith(FLASH_ROOT) for name in sources):
        raise BuildError("No Flash1017 implementation found in this checkout")
    for name in ("pyproject.toml", "setup.py", "README.md", "LICENSE"):
        sources[name] = checked_bytes(root / name)
    for pattern in ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*"):
        for path in root.glob(pattern):
            if path.is_file():
                sources[path.name] = checked_bytes(path)
    return dict(sorted(sources.items())), dict(sorted(package_paths.items()))


def validate_source_wheel(
    entries: dict[str, bytes], sources: dict[str, bytes], package_paths: dict[str, str]
) -> None:
    metadata = parse_metadata(entries[f"{DIST_INFO}/METADATA"])
    if (metadata.get("Name"), metadata.get("Version")) != ("omlx", VERSION):
        raise BuildError("Setuptools produced the wrong distribution identity")
    packaged = {name for name in entries if not name.startswith(DIST_INFO + "/")}
    if packaged != set(package_paths):
        missing, extra = sorted(set(package_paths) - packaged), sorted(
            packaged - set(package_paths)
        )
        raise BuildError(
            f"Source wheel package files differ: missing={missing}, extra={extra}"
        )
    for target, source in package_paths.items():
        if entries[target] != sources[source]:
            raise BuildError(f"Source wheel bytes differ from current source: {source}")
    if any(name.endswith((".so", ".metallib", ".pyc")) for name in entries):
        raise BuildError("The clean source wheel unexpectedly contains compiled files")


def merge_wheel(
    entries: dict[str, bytes],
    native: dict[str, bytes],
    license_text: bytes,
    provenance: dict,
) -> dict[str, bytes]:
    if set(native) != NATIVE_NAMES or set(native) & set(entries):
        raise BuildError(
            "Native files are missing, unexpected, or overlap current source wheel"
        )
    result = dict(entries)
    for suffix in ("RECORD", "RECORD.jws", "RECORD.p7s", "INSTALLER", "REQUESTED"):
        result.pop(f"{DIST_INFO}/{suffix}", None)
    result.update(native)
    result[f"{DIST_INFO}/licenses/upstream-native/LICENSE"] = license_text
    metadata = result[f"{DIST_INFO}/METADATA"].replace(b"\r\n", b"\n")
    headers, separator, body = metadata.partition(b"\n\n")
    if not separator:
        raise BuildError("Source METADATA has no header boundary")
    result[f"{DIST_INFO}/METADATA"] = (
        headers + b"\nLicense-File: upstream-native/LICENSE\n\n" + body
    )
    result[f"{DIST_INFO}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: flash1017-public-native-build (1)\n"
        f"Root-Is-Purelib: false\nTag: {TAG}\n\n"
    ).encode()
    result[f"{DIST_INFO}/FLASH1017-BUILD.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode()
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(result.items()):
        value = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        writer.writerow([name, "sha256=" + value, len(data)])
    writer.writerow([f"{DIST_INFO}/RECORD", "", ""])
    result[f"{DIST_INFO}/RECORD"] = output.getvalue().encode()
    return result


def write_wheel(path: Path, entries: dict[str, bytes]) -> None:
    # Constant timestamps, modes, compression and order avoid local path/mtime leaks.
    with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries, key=lambda n: (n.startswith(DIST_INFO + "/"), n)):
            info = ZipInfo(name, ZIP_TIME)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, entries[name], compresslevel=9)


def build(root: Path, base_wheel: Path, output_dir: Path) -> Path:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise BuildError("Build this cp312 artifact with CPython 3.12")
    for package, version in BUILD_TOOLS.items():
        if importlib.metadata.version(package) != version:
            raise BuildError(f"Use the pinned build tool {package}=={version}")
    native, license_text = load_base_assets(base_wheel)
    native_source_hashes = verify_native_sources(root)
    sources, package_paths = collect_sources(root)
    source_hashes = {name: digest(data) for name, data in sources.items()}
    # Check staged native Python/setup bytes too, so a change during collection
    # cannot slip through the earlier public-base check.
    for name in sources.keys() & native_source_hashes.keys():
        if digest(sources[name]) != native_source_hashes[name]:
            raise BuildError(f"Native input changed during collection: {name}")
    provenance = {
        "schema": "flash1017-public-native-build/1",
        "version": VERSION,
        "source_commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "source_files_sha256": source_hashes,
        "helper_sha256": digest(Path(__file__).read_bytes()),
        "base_commit": BASE_COMMIT,
        "base_wheel": {
            "url": BASE_URL,
            "sha256": BASE_WHEEL_SHA256,
            "advertised_tag": BASE_TAG,
        },
        "native_source_sha256": native_source_hashes,
        "native_assets_sha256": {name: digest(data) for name, data in native.items()},
        "build_tools": BUILD_TOOLS,
        "python": sys.version.split()[0],
        "output_tag": TAG,
        "historical_binary_identity_claim": False,
        "qualification": "packaging only; inference and performance require separate validation",
    }
    with tempfile.TemporaryDirectory(prefix="flash1017-build-") as temp:
        staging = Path(temp) / "source"
        stage_dist = Path(temp) / "wheel"
        stage_dist.mkdir()
        for relative, data in sources.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHONPATH",
                "PYTHONHOME",
                "SETUPTOOLS_SCM_PRETEND_VERSION",
                "CMAKE_ARGS",
                "OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET",
            }
        }
        env.update(
            OMLX_WITH_CUSTOM_KERNEL="0",
            SOURCE_DATE_EPOCH="315532800",
            PYTHONHASHSEED="0",
        )
        code = "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])"
        subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(stage_dist)],
            cwd=staging,
            env=env,
            check=True,
        )
        built = list(stage_dist.glob("*.whl"))
        if len(built) != 1:
            raise BuildError("Expected exactly one source wheel")
        entries = read_zip(built[0])
        validate_source_wheel(entries, sources, package_paths)
        result = merge_wheel(entries, native, license_text, provenance)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"omlx-{VERSION}-{TAG}.whl"
        write_wheel(output, result)
    checksum = digest(output.read_bytes())
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "wheel": str(output),
                "sha256": checksum,
                "source_files": len(sources),
                "native_assets": len(native),
                "tag": TAG,
            },
            indent=2,
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-wheel",
        type=Path,
        required=True,
        help="Downloaded public cp312 v0.7.1rc1 wheel; exact SHA required",
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dist/flash1017"))
    args = parser.parse_args()
    try:
        build(
            args.source_root.resolve(),
            args.base_wheel.resolve(),
            args.output_dir.resolve(),
        )
    except (
        BuildError,
        OSError,
        KeyError,
        subprocess.CalledProcessError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(f"Flash1017 wheel build refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
