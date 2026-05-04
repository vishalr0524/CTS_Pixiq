"""
Split annular UV crops into train/test structure then run PatchCore training.

Reads from:
    training/patchcore_uv/dataset/good/    ← annular crops from prepare_dataset_annular.py
    training/patchcore_uv/dataset/defect/  ← defect crops (optional, for evaluation only)

Creates anomalib MVTec-like structure via symlinks:
    training/patchcore_uv/dataset/split/
        train/good/     ← 80% of good images
        test/good/      ← 20% of good images
        test/defect/    ← all defect images (if --with-defect)

Then trains PatchCore using the existing train.py logic.

Usage:
    # Train on good images only
    uv run python training/patchcore_uv/split_and_train.py

    # Train + evaluate against defect images
    uv run python training/patchcore_uv/split_and_train.py --with-defect
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

os.environ["TRUST_REMOTE_CODE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GOOD_DIR    = "training/patchcore_uv/dataset/good_300"
DEFECT_DIR  = "training/patchcore_uv/dataset/defect"
SPLIT_DIR   = "training/patchcore_uv/dataset/split"
RESULTS_DIR = "training/patchcore_uv/results"

TRAIN_SPLIT = 0.80  # 80% train, 20% test


def _patch_anomalib_pandas3():
    """Monkey-patch anomalib for pandas 3.0 compatibility (same as train.py)."""
    import anomalib.data.datasets.image.folder as folder_mod
    from pandas import DataFrame

    def _patched_make_folder_dataset(
        normal_dir, root=None, abnormal_dir=None, normal_test_dir=None,
        mask_dir=None, split=None, extensions=None,
    ):
        from anomalib.data.utils.label import LabelName
        from anomalib.data.utils.split import Split
        from anomalib.data.utils.path import DirType, _prepare_files_labels, validate_and_resolve_path
        from collections.abc import Sequence

        def _resolve(path):
            if isinstance(path, Sequence) and not isinstance(path, str):
                return [validate_and_resolve_path(p, root) for p in path]
            return [validate_and_resolve_path(path, root)] if path is not None else []

        normal_dir_r   = _resolve(normal_dir)
        abnormal_dir_r = _resolve(abnormal_dir)
        normal_test_dir_r = _resolve(normal_test_dir)
        mask_dir_r     = _resolve(mask_dir)

        if len(normal_dir_r) == 0:
            raise ValueError("A folder location must be provided in normal_dir.")

        filenames, labels = [], []
        dirs = {DirType.NORMAL: normal_dir_r}
        if abnormal_dir_r:
            dirs[DirType.ABNORMAL] = abnormal_dir_r
        if normal_test_dir_r:
            dirs[DirType.NORMAL_TEST] = normal_test_dir_r
        if mask_dir_r:
            dirs[DirType.MASK] = mask_dir_r

        for dir_type, paths in dirs.items():
            for path in paths:
                fn, lb = _prepare_files_labels(path, dir_type, extensions)
                filenames += fn
                labels += [str(l) for l in lb]

        samples = DataFrame({"image_path": filenames, "label": labels})
        samples = samples.sort_values(by="image_path", ignore_index=True)

        NORMAL      = str(DirType.NORMAL)
        ABNORMAL    = str(DirType.ABNORMAL)
        NORMAL_TEST = str(DirType.NORMAL_TEST)
        MASK        = str(DirType.MASK)

        samples["label_index"] = int(LabelName.NORMAL)
        is_abnormal = samples.label == ABNORMAL
        samples.loc[is_abnormal, "label_index"] = int(LabelName.ABNORMAL)
        samples["label_index"] = samples["label_index"].astype("Int64")

        if len(mask_dir_r) > 0 and len(abnormal_dir_r) > 0:
            samples["mask_path"] = ""
            samples.loc[is_abnormal, "mask_path"] = samples.loc[
                samples.label == MASK
            ].image_path.to_numpy()
            samples["mask_path"] = samples["mask_path"].fillna("")
            samples = samples.astype({"mask_path": "str"})
        else:
            samples["mask_path"] = ""

        samples = samples.loc[
            (samples.label == NORMAL) |
            (samples.label == ABNORMAL) |
            (samples.label == NORMAL_TEST)
        ]
        samples = samples.astype({"image_path": "str"})

        samples["split"] = str(Split.TRAIN)
        samples.loc[
            (samples.label == ABNORMAL) | (samples.label == NORMAL_TEST),
            "split",
        ] = str(Split.TEST)

        samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

        if split:
            samples = samples[samples.split == str(split)]
            samples = samples.reset_index(drop=True)

        return samples

    folder_mod.make_folder_dataset = _patched_make_folder_dataset
    logger.info("Patched anomalib make_folder_dataset for pandas 3.0 compatibility")


def build_split_dir(good_images: list[Path], defect_images: list[Path], split_dir: Path) -> None:
    """Create anomalib-compatible train/test directory structure using symlinks.

    train/good/  ← TRAIN_SPLIT of good images
    test/good/   ← remaining good images
    test/defect/ ← all defect images (if any)
    """
    if split_dir.exists():
        shutil.rmtree(split_dir)

    train_good_dir  = split_dir / "train" / "good"
    test_good_dir   = split_dir / "test"  / "good"
    train_good_dir.mkdir(parents=True)
    test_good_dir.mkdir(parents=True)

    # Reproducible 80/20 split
    random.seed(42)
    shuffled = good_images.copy()
    random.shuffle(shuffled)
    split_idx  = int(len(shuffled) * TRAIN_SPLIT)
    train_imgs = shuffled[:split_idx]
    test_imgs  = shuffled[split_idx:]

    for img in train_imgs:
        os.symlink(img.resolve(), train_good_dir / img.name)
    for img in test_imgs:
        os.symlink(img.resolve(), test_good_dir / img.name)

    if defect_images:
        test_defect_dir = split_dir / "test" / "defect"
        test_defect_dir.mkdir(parents=True)
        for img in defect_images:
            os.symlink(img.resolve(), test_defect_dir / img.name)

    logger.info(f"  Train (good)  : {len(train_imgs)}")
    logger.info(f"  Test  (good)  : {len(test_imgs)}")
    logger.info(f"  Test  (defect): {len(defect_images)}")


def main():
    parser = argparse.ArgumentParser(description="Split UV dataset and train PatchCore")
    parser.add_argument("--with-defect", action="store_true",
                        help="Include defect images in test set for evaluation")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Randomly sample N good images (default: use all)")
    parser.add_argument("--backbone", default="wide_resnet50_2",
                        help="Feature extractor backbone (default: wide_resnet50_2)")
    parser.add_argument("--coreset-ratio", type=float, default=0.1,
                        help="Coreset sampling ratio (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    good_dir   = project_root / GOOD_DIR
    defect_dir = project_root / DEFECT_DIR
    split_dir  = project_root / SPLIT_DIR
    results_dir = project_root / RESULTS_DIR

    if not good_dir.exists() or not any(good_dir.iterdir()):
        logger.error(f"Good images not found: {good_dir}")
        logger.error("Run prepare_dataset_annular.py --good first.")
        sys.exit(1)

    good_images = sorted(good_dir.glob("*.png"))
    logger.info(f"Good images   : {len(good_images)} total")

    if args.max_images and args.max_images < len(good_images):
        random.seed(42)
        good_images = random.sample(good_images, args.max_images)
        good_images = sorted(good_images)
        logger.info(f"Good images   : {len(good_images)} (sampled)")

    defect_images = []
    if args.with_defect and defect_dir.exists():
        defect_images = sorted(defect_dir.glob("*.png"))
        logger.info(f"Defect images : {len(defect_images)}")
    elif args.with_defect:
        logger.warning(f"--with-defect set but {defect_dir} not found — skipping defect test set")

    # Build train/test split directory
    logger.info(f"\nBuilding split structure → {split_dir}")
    build_split_dir(good_images, defect_images, split_dir)

    # Import anomalib after patching
    try:
        import torch
        torch.set_float32_matmul_precision("medium")
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
        from anomalib.deploy import ExportType
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError:
        logger.error("anomalib not installed. Run: uv add anomalib")
        sys.exit(1)

    _patch_anomalib_pandas3()

    datamodule = Folder(
        name="uv_cone",
        root=str(split_dir),
        normal_dir="train/good",
        abnormal_dir="test/defect" if defect_images else None,
        normal_test_dir="test/good",
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
    )

    model = Patchcore(
        backbone=args.backbone,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=9,
    )

    logger.info(f"\nPatchCore config:")
    logger.info(f"  Backbone      : {args.backbone}")
    logger.info(f"  Layers        : layer2, layer3")
    logger.info(f"  Coreset ratio : {args.coreset_ratio}")
    logger.info(f"  Batch size    : {args.batch_size}")

    engine = Engine(default_root_dir=str(results_dir))

    logger.info("\nTraining PatchCore (1 epoch — memory bank build)...")
    engine.fit(model=model, datamodule=datamodule)

    logger.info("\nEvaluating...")
    engine.test(model=model, datamodule=datamodule)

    export_root = results_dir / "Patchcore" / "uv_cone" / "exported"
    logger.info(f"\nExporting Torch model → {export_root / 'torch'}")
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=str(export_root / "torch"),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model : {export_root / 'torch'}")
    logger.info(f"Deploy: cp -r {export_root}/torch/* models/patchcore_uv/weights/torch/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
