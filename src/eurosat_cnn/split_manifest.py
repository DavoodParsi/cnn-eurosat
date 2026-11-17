"""Create, rebuild and verify the train/val/test split from a manifest.

The project uses a fixed random split of the EuroSAT RGB dataset. The split
is recorded in ``data/split_manifest.csv`` with the columns ``filepath``,
``label`` and ``split``. ``filepath`` is relative to the dataset root, for
example ``train/AnnualCrop/AnnualCrop_1001.jpg``.

Command line (run from the repository root)::

    # Rebuild the split from the original EuroSAT RGB images
    python -m eurosat_cnn.split_manifest build --source <original_dataset_dir>

    # Check a split folder against the manifest
    python -m eurosat_cnn.split_manifest verify

    # Write a manifest from an existing split folder
    python -m eurosat_cnn.split_manifest create

``<original_dataset_dir>`` is the folder that directly contains the ten class
folders (AnnualCrop, Forest, ...). The source images are only read, never
modified.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")

MANIFEST_COLUMNS = ("filepath", "label", "split")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Defaults are relative to the repository root.
DEFAULT_DATASET_DIR = Path("data") / "EuroSAT_RGB"
DEFAULT_MANIFEST_PATH = Path("data") / "split_manifest.csv"


# --------------------------------------------
# Scan a split dataset folder
# --------------------------------------------

def _scan_images(dataset_root: Path) -> list[dict]:
    """Return one record per image found in ``dataset_root/<split>/<class>/``."""

    records = []

    for split in SPLITS:

        split_dir = dataset_root / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for path in sorted(split_dir.rglob("*")):

            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative = path.relative_to(dataset_root)

            # Images must sit inside a class folder; skip hidden entries.
            if len(relative.parts) < 3:
                continue

            if any(part.startswith(".") for part in relative.parts):
                continue

            records.append(
                {
                    "filepath": relative.as_posix(),
                    "label": relative.parts[1],
                    "split": split,
                }
            )

    return records


# --------------------------------------------
# Create a manifest
# --------------------------------------------

def create_split_manifest(dataset_root, manifest_path) -> int:
    """Write a manifest from an existing split folder. Returns the row count."""

    dataset_root = Path(dataset_root)
    records = _scan_images(dataset_root)

    if not records:
        raise ValueError(f"No images found in {dataset_root}")

    # Every split must contain the same classes.
    classes = {
        split: {r["label"] for r in records if r["split"] == split}
        for split in SPLITS
    }

    for split in SPLITS[1:]:

        if classes[split] != classes[SPLITS[0]]:

            raise ValueError(
                f"Class mismatch in '{split}': expected "
                f"{sorted(classes[SPLITS[0]])}, found {sorted(classes[split])}"
            )

    records.sort(
        key=lambda r: (SPLITS.index(r["split"]), r["label"], r["filepath"])
    )

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    return len(records)


# --------------------------------------------
# Read and validate a manifest
# --------------------------------------------

def read_manifest(manifest_path) -> list[dict]:
    """Read the manifest and validate every row."""

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        if not set(MANIFEST_COLUMNS).issubset(reader.fieldnames or []):

            raise ValueError(
                f"Manifest must contain the columns {list(MANIFEST_COLUMNS)}"
            )

        rows = [
            {key: (row[key] or "").strip() for key in MANIFEST_COLUMNS}
            for row in reader
        ]

    if not rows:
        raise ValueError("The manifest is empty.")

    seen = set()

    for number, row in enumerate(rows, start=2):

        relative = PurePosixPath(row["filepath"].replace("\\", "/"))
        parts = relative.parts

        is_valid = (
            row["split"] in SPLITS
            and len(parts) >= 3
            and parts[0] == row["split"]
            and parts[1] == row["label"]
            and not relative.is_absolute()
            and ".." not in parts
        )

        if not is_valid:
            raise ValueError(f"Invalid manifest row {number}: {row['filepath']}")

        row["filepath"] = relative.as_posix()

        if row["filepath"] in seen:
            raise ValueError(f"Duplicate path in manifest: {row['filepath']}")

        seen.add(row["filepath"])

    return rows


# --------------------------------------------
# Verify a split folder against the manifest
# --------------------------------------------

def verify_split_against_manifest(dataset_root, manifest_path) -> dict:
    """Compare a split folder with the manifest. Nothing is modified."""

    records = read_manifest(manifest_path)

    expected = {r["filepath"] for r in records}
    found = {r["filepath"] for r in _scan_images(Path(dataset_root))}

    missing = sorted(expected - found)
    extra = sorted(found - expected)

    return {
        "ok": not missing and not extra,
        "expected": len(expected),
        "found": len(found),
        "missing": missing,
        "extra": extra,
        "split_counts": dict(Counter(r["split"] for r in records)),
    }


# --------------------------------------------
# Rebuild the split from the original dataset
# --------------------------------------------

def build_split_from_manifest(
    source_root,
    manifest_path,
    output_root,
    *,
    overwrite: bool = False,
) -> dict:
    """Copy the original images into ``output_root/<split>/<class>/``.

    ``source_root`` must directly contain the class folders of the original
    dataset. Source files are only read. Images that already exist in the
    output are skipped unless ``overwrite`` is True.
    """

    source_root = Path(source_root)
    output_root = Path(output_root)

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    if source_root.resolve() == output_root.resolve():
        raise ValueError("The output folder must differ from the source folder.")

    records = read_manifest(manifest_path)

    plan = []
    missing = []

    for record in records:

        parts = PurePosixPath(record["filepath"]).parts
        source = source_root / record["label"] / parts[-1]

        if not source.is_file():
            missing.append(source)

        plan.append((source, output_root.joinpath(*parts)))

    # Fail before copying anything if some source images are not found.
    if missing:

        raise FileNotFoundError(
            f"{len(missing):,} images listed in the manifest were not found "
            f"under {source_root} (first missing: {missing[0]}). Expected "
            "layout: <source>/<Class>/<Class>_<number>.jpg"
        )

    copied = 0
    skipped = 0

    for source, destination in plan:

        if destination.exists() and not overwrite:
            skipped += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    report = verify_split_against_manifest(output_root, manifest_path)
    report.update({"copied": copied, "skipped": skipped})

    return report


# --------------------------------------------
# Command line interface
# --------------------------------------------

def _print_report(report: dict) -> None:

    print(f"Expected images : {report['expected']:,}")
    print(f"Found images    : {report['found']:,}")
    print(f"Missing         : {len(report['missing']):,}")
    print(f"Extra           : {len(report['extra']):,}")

    if "copied" in report:
        print(f"Copied          : {report['copied']:,}")
        print(f"Skipped         : {report['skipped']:,}")

    for name in ("missing", "extra"):
        for path in report[name][:10]:
            print(f"  {name}: {path}")

    sizes = ", ".join(
        f"{split}={report['split_counts'].get(split, 0):,}" for split in SPLITS
    )

    print(f"Split sizes (manifest): {sizes}")

    if report["ok"]:
        print("RESULT: the dataset matches the manifest.")
    else:
        print("RESULT: the dataset does NOT match the manifest.")


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="rebuild the split from the original dataset")
    build.add_argument("--source", required=True, help="folder containing the original class folders")
    build.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    build.add_argument("--output", default=str(DEFAULT_DATASET_DIR))
    build.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser("verify", help="check a split folder against the manifest")
    verify.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))

    create = commands.add_parser("create", help="write a manifest from a split folder")
    create.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    create.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))

    args = parser.parse_args(argv)

    if args.command == "create":

        count = create_split_manifest(args.dataset, args.manifest)
        print(f"Manifest written: {args.manifest} ({count:,} images)")
        return 0

    if args.command == "verify":

        report = verify_split_against_manifest(args.dataset, args.manifest)

    else:

        report = build_split_from_manifest(
            args.source,
            args.manifest,
            args.output,
            overwrite=args.overwrite,
        )

    _print_report(report)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())