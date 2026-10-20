"""Source distribution checks must reject missing or changed runtime assets."""

import contextlib
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.check_distribution import main


class DistributionTests(unittest.TestCase):
    def check_archive(self, *, manifest=True, asset=b"runner", expected=b"runner"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "release.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                entries = {}
                if manifest:
                    entries["dependencies/ASSETS.json"] = json.dumps(
                        {"third_party/runner.py": hashlib.sha256(expected).hexdigest()}
                    ).encode()
                if asset is not None:
                    entries["third_party/runner.py"] = asset
                for name, body in entries.items():
                    member = tarfile.TarInfo("harness2-0.1.0/" + name)
                    member.size = len(body)
                    archive.addfile(member, io.BytesIO(body))
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return main([str(path)])

    def test_valid_archive(self):
        self.assertEqual(self.check_archive(), 0)

    def test_missing_manifest(self):
        with self.assertRaises(SystemExit) as error:
            self.check_archive(manifest=False)
        self.assertEqual(error.exception.code, 1)

    def test_missing_asset(self):
        with self.assertRaises(SystemExit) as error:
            self.check_archive(asset=None)
        self.assertEqual(error.exception.code, 1)

    def test_changed_asset(self):
        with self.assertRaises(SystemExit) as error:
            self.check_archive(asset=b"changed")
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
