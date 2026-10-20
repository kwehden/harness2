#!/usr/bin/env python3
"""Check that a built source distribution includes every benchmark release asset."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    with tarfile.open(args.archive, "r:gz") as archive:
        manifests = [m for m in archive.getnames() if m.endswith("/dependencies/ASSETS.json")]
        if len(manifests) != 1:
            parser.exit(1, "Source archive must include one dependencies/ASSETS.json\n")
        manifest = manifests[0]
        prefix = manifest.removesuffix("dependencies/ASSETS.json")
        expected = json.load(archive.extractfile(manifest))
        failures = []
        for name, digest in expected.items():
            try:
                source = archive.extractfile(prefix + name)
            except KeyError:
                source = None
            if source is None or hashlib.sha256(source.read()).hexdigest() != digest:
                failures.append(name)
        if failures:
            parser.exit(
                1, "Missing or changed source-distribution assets: " + ", ".join(failures) + "\n"
            )
    print(f"PASS: {len(expected)} source-distribution asset checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
