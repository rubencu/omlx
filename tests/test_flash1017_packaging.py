# SPDX-License-Identifier: Apache-2.0
"""CPU-only packaging contracts; run directly to avoid GPU pytest conftest."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_flash1017_wheel.py"
_SPEC = importlib.util.spec_from_file_location("flash1017_packaging", _SCRIPT)
pack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pack)


def bundle(minimum=15 << 16, cpu=0x0100000C):
    header = struct.pack("<8I", 0xFEEDFACF, cpu, 0, 8, 1, 24, 0, 0)
    return header + struct.pack("<6I", 0x32, 24, 1, minimum, minimum, 0)


def native_assets():
    return {
        name: bundle() if name.endswith(".so") else b"metal fixture"
        for name in pack.NATIVE_NAMES
    }


def source_entries():
    return {
        "omlx/_version.py": f'__version__ = "{pack.VERSION}"\n'.encode(),
        "omlx_web/static/app.js": b"current fork web content\n",
        f"{pack.DIST_INFO}/METADATA": (
            f"Metadata-Version: 2.4\nName: omlx\nVersion: {pack.VERSION}\n"
            "License-File: LICENSE\n\nCurrent fork description.\n"
        ).encode(),
        f"{pack.DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        f"{pack.DIST_INFO}/licenses/LICENSE": b"current license",
        f"{pack.DIST_INFO}/RECORD": b"stale record",
    }


class NativeInputTests(unittest.TestCase):
    def test_arm64_macos15_bundle_is_accepted(self):
        pack.check_arm64_bundle(bundle(), "fixture.so")

    def test_x86_or_higher_minimum_is_rejected(self):
        for data in (bundle(cpu=0x01000007), bundle(minimum=16 << 16), b"short"):
            with self.subTest(data=data), self.assertRaises(pack.BuildError):
                pack.check_arm64_bundle(data, "fixture.so")

    def test_corrupt_load_command_is_rejected(self):
        data = bytearray(bundle())
        struct.pack_into("<I", data, 36, 0)
        with self.assertRaises(pack.BuildError):
            pack.check_arm64_bundle(bytes(data), "fixture.so")

    def test_public_wheel_hash_is_required_before_reading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "base.whl"
            path.write_bytes(b"not the pinned wheel")
            with self.assertRaisesRegex(pack.BuildError, "SHA-256"):
                pack.load_base_assets(path)

    def test_exact_native_allowlist_and_license_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "base.whl"
            entries = native_assets()
            entries.update(
                {
                    f"{pack.BASE_DIST_INFO}/METADATA": f"Name: omlx\nVersion: {pack.BASE_VERSION}\n\n".encode(),
                    f"{pack.BASE_DIST_INFO}/WHEEL": f"Root-Is-Purelib: false\nTag: {pack.BASE_TAG}\n\n".encode(),
                    f"{pack.BASE_DIST_INFO}/licenses/LICENSE": b"upstream native license",
                    "omlx/stale_upstream.py": b"must never be returned",
                }
            )
            with ZipFile(path, "w") as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)
            with patch.object(
                pack, "BASE_WHEEL_SHA256", pack.digest(path.read_bytes())
            ):
                native, license_text = pack.load_base_assets(path)
            self.assertEqual(native, native_assets())
            self.assertEqual(license_text, b"upstream native license")

    def test_extra_or_missing_native_asset_is_rejected(self):
        for delta in ("extra", "missing"):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "base.whl"
                entries = native_assets()
                if delta == "extra":
                    entries["omlx/unexpected.dylib"] = b"extra"
                else:
                    entries.pop(next(iter(entries)))
                entries[f"{pack.BASE_DIST_INFO}/METADATA"] = (
                    f"Name: omlx\nVersion: {pack.BASE_VERSION}\n\n".encode()
                )
                entries[f"{pack.BASE_DIST_INFO}/WHEEL"] = (
                    f"Root-Is-Purelib: false\nTag: {pack.BASE_TAG}\n\n".encode()
                )
                with ZipFile(path, "w") as archive:
                    for name, data in entries.items():
                        archive.writestr(name, data)
                with (
                    patch.object(
                        pack, "BASE_WHEEL_SHA256", pack.digest(path.read_bytes())
                    ),
                    self.assertRaisesRegex(pack.BuildError, "allowlist"),
                ):
                    pack.load_base_assets(path)

    def test_changed_added_and_deleted_native_sources_fail(self):
        baseline = {
            "setup.py": b"setup()",
            pack.NATIVE_ROOT + "common/csrc/kernel.h": b"original header",
        }
        for alteration in ("same", "changed", "added", "deleted"):
            with (
                self.subTest(alteration=alteration),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                for name, data in baseline.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                if alteration == "changed":
                    (root / "setup.py").write_bytes(b"changed build flags")
                if alteration == "added":
                    (root / pack.NATIVE_ROOT / "new.metal").write_bytes(b"new kernel")
                if alteration == "deleted":
                    (root / pack.NATIVE_ROOT / "common/csrc/kernel.h").unlink()
                if alteration == "same":
                    self.assertEqual(
                        set(pack.check_native_snapshot(root, baseline)), set(baseline)
                    )
                else:
                    with self.assertRaises(pack.BuildError):
                        pack.check_native_snapshot(root, baseline)


class SourceCollectionTests(unittest.TestCase):
    def make_tree(self, root, data_declared=True):
        declarations = (
            '[tool.setuptools.package-data]\n"omlx.patches.deepseek_v41.flash1017" = ["*.json", "NOTICE"]\n'
            if data_declared
            else ""
        )
        files = {
            "setup.py": b"from setuptools import setup\nsetup()\n",
            "pyproject.toml": (
                '[build-system]\nbuild-backend = "setuptools.build_meta"\n'
                '[project]\ndependencies = ["mlx==0.32.4"]\n' + declarations
            ).encode(),
            "README.md": b"current fork readme",
            "LICENSE": b"current license",
            "omlx/_version.py": f'__version__ = "{pack.VERSION}"\n'.encode(),
            pack.FLASH_ROOT + "__init__.py": b"# current fork\n",
            pack.FLASH_ROOT + "profile.json": b'{"frozen": true}',
            pack.FLASH_ROOT + "NOTICE": b"retain attribution",
            "apps/omlx-web/omlx_web/__init__.py": b"# current web\n",
        }
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return files

    def test_sources_and_declared_data_come_from_current_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = self.make_tree(root)
            sources, package_paths = pack.collect_sources(root)
            self.assertEqual(sources, expected)
            self.assertEqual(
                package_paths["omlx_web/__init__.py"],
                "apps/omlx-web/omlx_web/__init__.py",
            )
            self.assertIn(pack.FLASH_ROOT + "profile.json", sources)

    def test_missing_flash_asset_declaration_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root, data_declared=False)
            with self.assertRaisesRegex(pack.BuildError, "package-data"):
                pack.collect_sources(root)

    def test_local_native_and_bytecode_are_not_source_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            stale = root / next(iter(pack.NATIVE_NAMES))
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"untrusted local binary")
            (root / "omlx/__pycache__").mkdir()
            (root / "omlx/__pycache__/stale.pyc").write_bytes(b"bytecode")
            sources, _ = pack.collect_sources(root)
            self.assertFalse(
                any(name.endswith((".so", ".metallib", ".pyc")) for name in sources)
            )

    def test_wrong_fork_version_or_mlx_pin_is_fatal(self):
        for target, value in (
            ("omlx/_version.py", b'__version__ = "0.7.1rc1"\n'),
            ("pyproject.toml", None),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_tree(root)
                path = root / target
                path.write_bytes(
                    value
                    if value is not None
                    else path.read_bytes().replace(b"mlx==0.32.4", b"mlx>=0.32.4")
                )
                with self.assertRaises(pack.BuildError):
                    pack.collect_sources(root)

    def test_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            (root / "omlx/linked.py").symlink_to(root / "omlx/_version.py")
            with self.assertRaisesRegex(pack.BuildError, "symlink"):
                pack.collect_sources(root)

    def test_missing_extra_or_stale_python_in_source_wheel_is_fatal(self):
        pristine = source_entries()
        package_paths = {
            "omlx/_version.py": "omlx/_version.py",
            "omlx_web/static/app.js": "apps/omlx-web/omlx_web/static/app.js",
        }
        sources = {source: pristine[target] for target, source in package_paths.items()}
        pack.validate_source_wheel(pristine, sources, package_paths)
        for change in ("missing", "extra", "stale"):
            with self.subTest(change=change):
                entries = dict(pristine)
                if change == "missing":
                    entries.pop("omlx_web/static/app.js")
                elif change == "extra":
                    entries["omlx/upstream_stale.py"] = b"stale module"
                else:
                    entries["omlx/_version.py"] = b"stale module"
                with self.assertRaises(pack.BuildError):
                    pack.validate_source_wheel(entries, sources, package_paths)


class WheelOutputTests(unittest.TestCase):
    def test_nonpure_tags_version_licenses_and_complete_record(self):
        native = native_assets()
        result = pack.merge_wheel(
            source_entries(), native, b"original upstream license", {"test": True}
        )
        wheel = pack.parse_metadata(result[f"{pack.DIST_INFO}/WHEEL"])
        self.assertEqual(wheel["Root-Is-Purelib"], "false")
        self.assertEqual(wheel.get_all("Tag"), ["cp312-cp312-macosx_15_0_arm64"])
        metadata = pack.parse_metadata(result[f"{pack.DIST_INFO}/METADATA"])
        self.assertEqual(metadata["Version"], pack.VERSION)
        self.assertEqual(
            metadata.get_all("License-File"), ["LICENSE", "upstream-native/LICENSE"]
        )
        self.assertEqual(
            result[f"{pack.DIST_INFO}/licenses/upstream-native/LICENSE"],
            b"original upstream license",
        )
        self.assertEqual({name: result[name] for name in native}, native)
        rows = list(
            csv.reader(io.StringIO(result[f"{pack.DIST_INFO}/RECORD"].decode()))
        )
        self.assertEqual({row[0] for row in rows}, set(result))
        self.assertEqual(len(rows), len(result))
        for name, hash_value, size in rows:
            if name.endswith("/RECORD"):
                self.assertEqual((hash_value, size), ("", ""))
            else:
                expected = (
                    base64.urlsafe_b64encode(hashlib.sha256(result[name]).digest())
                    .rstrip(b"=")
                    .decode()
                )
                self.assertEqual(hash_value, "sha256=" + expected)
                self.assertEqual(int(size), len(result[name]))
        self.assertEqual(
            json.loads(result[f"{pack.DIST_INFO}/FLASH1017-BUILD.json"]), {"test": True}
        )

    def test_native_overlap_is_fatal(self):
        entries = source_entries()
        entries[next(iter(pack.NATIVE_NAMES))] = b"stale local binary"
        with self.assertRaises(pack.BuildError):
            pack.merge_wheel(entries, native_assets(), b"license", {})

    def test_archive_bytes_are_deterministic_and_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = pack.merge_wheel(
                source_entries(), native_assets(), b"license", {}
            )
            one, two = root / "one.whl", root / "two.whl"
            pack.write_wheel(one, entries)
            pack.write_wheel(two, dict(reversed(list(entries.items()))))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            self.assertEqual(pack.read_zip(one), entries)
            with self.assertRaises(FileExistsError):
                pack.write_wheel(one, entries)

    def test_unsafe_zip_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.whl"
            with ZipFile(path, "w") as archive:
                archive.writestr("../outside.py", b"bad")
            with self.assertRaises(pack.BuildError):
                pack.read_zip(path)


if __name__ == "__main__":
    unittest.main()
