"""
Train PatchCore model for UV thread mixup detection.

Uses anomalib's PatchCore implementation trained on good (normal)
unwrapped UV cone images. Defects are detected as anomalies.

Includes monkey-patches for anomalib 2.2.0 + pandas 3.0 compatibility.

Usage:
    # Train PatchCore (good images only, 80/20 split)
    uv run python training/patchcore_uv/train.py

    # Train with defect test set
    uv run python training/patchcore_uv/train.py --with-defect
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

# Required for anomalib/timm model loading
os.environ["TRUST_REMOTE_CODE"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patch_anomalib_pandas3():
    """Monkey-patch anomalib for pandas 3.0 compatibility.

    pandas 3.0 breaks two things in make_folder_dataset:
    1. df.loc[mask, "new_col"] = scalar — rejected when column doesn't exist
    2. df.label == DirType.NORMAL — str Enum comparison returns all False

    We rewrite make_folder_dataset using plain string comparisons and
    pre-initialized columns.
    """
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

        normal_dir_r = _resolve(normal_dir)
        abnormal_dir_r = _resolve(abnormal_dir)
        normal_test_dir_r = _resolve(normal_test_dir)
        mask_dir_r = _resolve(mask_dir)

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
                # Store plain strings, not enum objects (pandas 3.0 fix)
                filenames += fn
                labels += [str(l) for l in lb]

        samples = DataFrame({"image_path": filenames, "label": labels})
        samples = samples.sort_values(by="image_path", ignore_index=True)

        # Use plain string values for all comparisons (pandas 3.0 fix)
        NORMAL = str(DirType.NORMAL)           # "normal"
        ABNORMAL = str(DirType.ABNORMAL)       # "abnormal"
        NORMAL_TEST = str(DirType.NORMAL_TEST) # "normal_test"
        MASK = str(DirType.MASK)               # "mask_dir"

        # Pre-initialize columns (pandas 3.0 fix)
        samples["label_index"] = int(LabelName.NORMAL)
        is_abnormal = samples.label == ABNORMAL
        samples.loc[is_abnormal, "label_index"] = int(LabelName.ABNORMAL)
        samples["label_index"] = samples["label_index"].astype("Int64")

        # Mask path
        if len(mask_dir_r) > 0 and len(abnormal_dir_r) > 0:
            samples["mask_path"] = ""
            samples.loc[is_abnormal, "mask_path"] = samples.loc[
                samples.label == MASK
            ].image_path.to_numpy()
            samples["mask_path"] = samples["mask_path"].fillna("")
            samples = samples.astype({"mask_path": "str"})
        else:
            samples["mask_path"] = ""

        # Remove mask rows
        samples = samples.loc[
            (samples.label == NORMAL) |
            (samples.label == ABNORMAL) |
            (samples.label == NORMAL_TEST)
        ]
        samples = samples.astype({"image_path": "str"})

        # Pre-initialize split column with plain strings (pandas 3.0 fix)
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


def setup_directory_structure(dataset_dir, output_dir, good_images, defect_images):
    """Create anomalib-compatible train/test directory with symlinks.

    anomalib Folder with test_split_mode=from_dir expects:
        root/train/good/    <- normal training images
        root/test/good/     <- normal test images
        root/test/defect/   <- anomalous test images (optional)
    """
    anomalib_root = output_dir / "anomalib_data"
    if anomalib_root.exists():
        shutil.rmtree(anomalib_root)

    train_dir = anomalib_root / "train" / "good"
    test_good_dir = anomalib_root / "test" / "good"
    train_dir.mkdir(parents=True)
    test_good_dir.mkdir(parents=True)

    # 80/20 split
    random.seed(42)
    shuffled = good_images.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_imgs = shuffled[:split_idx]
    test_imgs = shuffled[split_idx:]

    for img in train_imgs:
        os.symlink(img.resolve(), train_dir / img.name)
    for img in test_imgs:
        os.symlink(img.resolve(), test_good_dir / img.name)

    if defect_images:
        test_defect_dir = anomalib_root / "test" / "defect"
        test_defect_dir.mkdir(parents=True)
        for img in defect_images:
            os.symlink(img.resolve(), test_defect_dir / img.name)

    logger.info(f"  Train (good): {len(train_imgs)}")
    logger.info(f"  Test (good): {len(test_imgs)}")
    logger.info(f"  Test (defect): {len(defect_images)}")

    return anomalib_root


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore for UV thread mixup")
    parser.add_argument(
        "--dataset", type=str, default="training/patchcore_uv/dataset",
        help="Dataset directory (with good/ and optionally defect/ subdirs)",
    )
    parser.add_argument(
        "--output", type=str, default="training/patchcore_uv/results",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--with-defect", action="store_true",
        help="Include defect images in test set",
    )
    parser.add_argument(
        "--backbone", type=str, default="wide_resnet50_2",
        help="Backbone model (wide_resnet50_2, resnet18, etc.)",
    )
    parser.add_argument(
        "--layers", type=str, default="layer2,layer3",
        help="Backbone layers to use (comma-separated)",
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.1,
        help="Coreset sampling ratio",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    good_dir = dataset_dir / "good"
    defect_dir = dataset_dir / "defect"

    if not good_dir.exists():
        logger.error(f"Good directory not found: {good_dir}")
        logger.info("Run prepare_dataset.py first")
        return

    good_images = list(good_dir.glob("*.png"))
    logger.info(f"Found {len(good_images)} good images")

    defect_images = []
    if args.with_defect and defect_dir.exists():
        defect_images = list(defect_dir.glob("*.png"))
        if defect_images:
            logger.info(f"Found {len(defect_images)} defect images for testing")

    # Import anomalib
    try:
        import torch
        torch.set_float32_matmul_precision('medium')

        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError:
        logger.error("anomalib not installed. Run: uv add anomalib")
        return

    # Apply pandas 3.0 compatibility patch
    _patch_anomalib_pandas3()

    # Create anomalib directory structure with pre-split symlinks
    anomalib_root = setup_directory_structure(
        dataset_dir, output_dir, good_images, defect_images,
    )

    from torchvision.transforms.v2 import Resize
    resize_transform = Resize((256, 720))

    # Use pre-split directories — all paths relative to root
    # make_folder_dataset assigns split=TRAIN to normal_dir, split=TEST to
    # normal_test_dir and abnormal_dir, then filters by split param.
    datamodule = Folder(
        name="uv_thread_mixup",
        root=str(anomalib_root),
        normal_dir="train/good",
        abnormal_dir="test/defect" if defect_images else None,
        normal_test_dir="test/good",
        train_batch_size=8,
        eval_batch_size=1,
        num_workers=4,
        augmentations=resize_transform,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )

    # Create model
    layers = [l.strip() for l in args.layers.split(",")]
    model = Patchcore(
        backbone=args.backbone,
        layers=layers,
        coreset_sampling_ratio=args.coreset_ratio,
    )

    engine = Engine(
        default_root_dir=str(output_dir),
        max_epochs=1,  # PatchCore only needs 1 epoch
    )

    logger.info("Training PatchCore model...")
    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Good images: {len(good_images)}")
    logger.info(f"  Defect images: {len(defect_images)}")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Layers: {args.layers}")
    logger.info(f"  Coreset ratio: {args.coreset_ratio}")
    logger.info(f"  Output: {output_dir}")

    engine.fit(model=model, datamodule=datamodule)

    logger.info("Testing model...")
    engine.test(model=model, datamodule=datamodule)

    # Export model
    logger.info("Exporting model...")
    export_dir = output_dir / "exported"
    engine.export(
        model=model,
        export_type="torch",
        export_root=str(export_dir),
    )

    logger.info(f"\nTraining complete!")
    logger.info(f"Model saved to: {export_dir}")
    logger.info(f"\nTo deploy: cp -r {export_dir}/torch/* models/patchcore_uv/weights/torch/")


if __name__ == "__main__":
    main()
