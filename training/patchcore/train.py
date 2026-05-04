"""
Train PatchCore model on annular (donut) cone crops using anomalib.

Prerequisites:
    Run prepare_dataset.py first to generate donut cone crops in dataset/.

Dataset must be in MVTec-like format:
    dataset/
        train/good/     ← normal donut crops for training (80%)
        test/good/      ← normal donut crops for testing (20%)
        test/stain/     ← stain donut crops for testing (optional)

Includes monkey-patches for anomalib 2.2.0 + pandas 3.0 compatibility.

Usage:
    # Basic training (normals only)
    uv run python training/patchcore/train.py

    # With stain images for validation
    uv run python training/patchcore/train.py --with-stain

    # Custom paths
    uv run python training/patchcore/train.py \\
        --dataset training/patchcore/dataset \\
        --output training/patchcore/results
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

os.environ["TRUST_REMOTE_CODE"] = "1"

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
                filenames += fn
                labels += [str(l) for l in lb]

        samples = DataFrame({"image_path": filenames, "label": labels})
        samples = samples.sort_values(by="image_path", ignore_index=True)

        NORMAL = str(DirType.NORMAL)
        ABNORMAL = str(DirType.ABNORMAL)
        NORMAL_TEST = str(DirType.NORMAL_TEST)
        MASK = str(DirType.MASK)

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


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on annular (donut) cone crops")
    parser.add_argument(
        "--dataset", default="training/patchcore/dataset",
        help="Root directory with train/good/ and test/good/ subdirectories",
    )
    parser.add_argument(
        "--output", default="training/patchcore/results",
        help="Output directory for trained model and logs",
    )
    parser.add_argument(
        "--backbone", default="wide_resnet50_2",
        help="Feature extractor backbone (default: wide_resnet50_2)",
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.1,
        help="Coreset sampling ratio (default: 0.1)",
    )
    parser.add_argument(
        "--with-stain", action="store_true",
        help="Include test/stain/ directory as abnormal test data",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--export-torch", action="store_true", default=True,
        help="Export model as Torch checkpoint (default: True)",
    )
    parser.add_argument(
        "--export-openvino", action="store_true",
        help="Also export as OpenVINO IR for faster inference",
    )
    parser.add_argument(
        "--name", default="cone_surface",
        help="Dataset name (used for output directory structure)",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    dataset_dir = Path(args.dataset)
    train_good = dataset_dir / "train" / "good"
    test_good = dataset_dir / "test" / "good"
    test_stain = dataset_dir / "test" / "stain"

    if not train_good.exists():
        print(f"ERROR: Training directory not found: {train_good}")
        print("Expected MVTec-like structure: dataset/train/good/")
        sys.exit(1)

    num_train = len(list(train_good.glob("*")))
    num_test = len(list(test_good.glob("*"))) if test_good.exists() else 0
    print(f"Train images:  {num_train} in {train_good}")
    print(f"Test (normal): {num_test} in {test_good}")

    if num_train < 10:
        print("ERROR: Need at least 10 normal images for PatchCore training.")
        sys.exit(1)

    # Check for stain images
    abnormal_dir = None
    has_stain = test_stain.exists() and any(test_stain.iterdir())
    if args.with_stain and has_stain:
        num_stain = len(list(test_stain.glob("*")))
        print(f"Test (stain):  {num_stain} in {test_stain}")
        abnormal_dir = "stain"
    elif args.with_stain:
        print(f"WARNING: --with-stain specified but {test_stain} is empty or missing.")

    # Patch anomalib for pandas 3.0 compatibility before importing
    _patch_anomalib_pandas3()

    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # Configure dataset — MVTec-like structure with pre-split directories
    datamodule = Folder(
        name=args.name,
        root=str(dataset_dir),
        normal_dir="train/good",
        abnormal_dir=abnormal_dir,
        normal_test_dir="test/good",
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
    )

    # Configure PatchCore model
    model = Patchcore(
        backbone=args.backbone,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=9,
    )

    print(f"\nModel: PatchCore")
    print(f"  Backbone: {args.backbone}")
    print(f"  Layers: layer2, layer3")
    print(f"  Coreset ratio: {args.coreset_ratio}")

    # Configure training engine
    engine = Engine(
        default_root_dir=args.output,
    )

    # Train (PatchCore only needs 1 epoch — it builds a memory bank, not a gradient-trained model)
    print(f"\nTraining PatchCore...")
    engine.fit(model=model, datamodule=datamodule)

    # Test (evaluate on held-out normals + stains if available)
    print(f"\nEvaluating...")
    engine.test(model=model, datamodule=datamodule)

    # Export model
    export_root = Path(args.output) / "Patchcore" / args.name / "exported"

    if args.export_torch:
        print(f"\nExporting Torch model...")
        engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=str(export_root / "torch"),
        )
        print(f"  Saved: {export_root / 'torch'}")

    if args.export_openvino:
        print(f"\nExporting OpenVINO model...")
        engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=str(export_root / "openvino"),
        )
        print(f"  Saved: {export_root / 'openvino'}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Results:  {args.output}")
    print(f"Exported: {export_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
