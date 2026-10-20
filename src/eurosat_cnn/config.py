"""Project-wide constants and filesystem paths."""

import os
from dataclasses import dataclass
from pathlib import Path

# --- Constants (values only, no actions) ---
PROJECT_NAME = "CNN_Image_Classification"
PROJECT_VERSION = "1.0"

SEED = 42
IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 32
DATASET_NAME = "EuroSAT_RGB"


def get_project_root() -> Path:
    """Return the project root.

    Defaults to the repository root, found relative to this file
    (src/eurosat_cnn/config.py -> three levels up is the root).
    Set EUROSAT_PROJECT_ROOT to override, e.g. on Google Colab to
    point at a folder on Google Drive.
    """
    override = os.environ.get("EUROSAT_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """All filesystem paths for one model, in one place."""

    root: Path
    dataset_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    model_output_dir: Path
    checkpoint_dir: Path
    reports_dir: Path
    model_dir: Path
    plots_dir: Path

    def create_output_dirs(self) -> None:
        """Create output folders. Called explicitly, never on import."""
        for directory in (
            self.model_output_dir,
            self.checkpoint_dir,
            self.reports_dir,
            self.model_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_paths(model_name: str) -> ProjectPaths:
    """Build all paths for the given model ("high" | "low" | ...)."""
    root = get_project_root()
    dataset_dir = root / "data" / DATASET_NAME
    model_output_dir = root / "outputs" / model_name

    return ProjectPaths(
        root=root,
        dataset_dir=dataset_dir,
        train_dir=dataset_dir / "train",
        val_dir=dataset_dir / "val",
        test_dir=dataset_dir / "test",
        model_output_dir=model_output_dir,
        checkpoint_dir=model_output_dir / "checkpoints",
        reports_dir=model_output_dir / "reports",
        model_dir=model_output_dir / "models",
        plots_dir=model_output_dir / "plots",
    )