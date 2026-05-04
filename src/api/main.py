"""
FastAPI application - main entry point.

Endpoints match old Flask API names for frontend compatibility:
    /tube           - Tube pattern teaching
    /stain          - Stain baseline teaching
    /extract        - Cone dimension extraction
    /color_detection - Color detection teaching
    /delete_master  - Delete teaching data
    /get_teaching_data - List all taught materials
    /tube_ocr       - OCR data (placeholder)
    /get_tube_img   - Get processed tube image
    /restart        - Restart service
    /shutdown       - Shutdown service

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 5002 --reload

    Or:
    uv run python run_api.py --port 5002
"""

import base64
import json
import sqlite3 as _sqlite3
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.auth_router import router as auth_router
from auth.service import AuthService
from db.schema import init_db as init_auth_db

from api.models import (
    TubeTeachRequest, TubeTeachResponse,
    StainDetectRequest, StainDetectResponse, StainDetectResult,
    ColorDetectRequest, ColorDetectResponse,
    DeleteMasterRequest, DeleteMasterResponse,
    GetTeachingDataResponse, TeachingDataItem,
    TubeOCRRequest, TubeOCRResponse,
    ExtractRequest, ExtractResponse,
    InspectRequest, InspectResponse,
    RetrainResult, RetrainAllResponse,
    RecipeRequest, RecipeResponse, RecipeListResponse, MasterListResponse,
    StatusResponse, RestartResponse, ShutdownResponse,
    CameraHealthStatus, PLCHealthStatus,
    SystemHealthResponse, CameraHealthResponse, PLCHealthResponse,
)

# Centralized logging
from logging_config import (
    setup_logging,
    get_logger,
    PerformanceLogger,
    event_logger,
    set_correlation_id,
)

# Initialize logging (will be reconfigured on startup with config)
logger = get_logger(__name__)

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="GHCL Yarn Cone Inspection API",
    description="API for teaching and inspection operations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth router
app.include_router(auth_router)


# Request context middleware for correlation ID
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add correlation ID to each request for tracing."""
    # Get correlation ID from header or generate new one
    correlation_id = request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # Log request
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "http_method": request.method,
            "http_path": request.url.path,
            "client_ip": request.client.host if request.client else None,
        }
    )

    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Log response
    logger.info(
        f"{request.method} {request.url.path} completed",
        extra={
            "http_method": request.method,
            "http_path": request.url.path,
            "http_status": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }
    )

    return response


# Global state
_start_time = time.time()
_inspector = None
_teacher = None
_recipe_store = None
_config = None


# ============================================================================
# Startup / Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize inspection modules on startup."""
    global _inspector, _teacher, _config

    # Load config (src/config.json)
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            _config = json.load(f)
    else:
        _config = {
            "inspection": {
                "weights": {"visible": "weights/visible_yolo.pt"},
                "database": "materials.db",
                "pixels_per_mm": 5.0,
                "yolo_conf": 0.6,
            },
            "teaching": {
                "template_dir": "templates/tube",
            },
            "logging": {
                "level": "INFO",
                "directory": "logs",
                "json_logs": True,
            }
        }

    # Initialize centralized logging with config
    setup_logging(
        config=_config,
        service_name="sieger-teaching-api",
        log_dir=_config.get("logging", {}).get("directory", "logs"),
    )

    logger.info(
        "Teaching API starting",
        extra={
            "config_path": str(config_path),
            "config_loaded": config_path.exists(),
        }
    )

    # Initialize auth — uses same sieger.db
    data_root = _config.get("data_root", "sieger_data")
    db_path = str(Path(data_root) / "sieger.db")
    auth_conn = init_auth_db(db_path)
    session_hours = float(_config.get("auth", {}).get("session_hours", 8.0))
    auth_service = AuthService(auth_conn, default_session_hours=session_hours)
    auth_service.seed_admin(
        username=_config.get("auth", {}).get("admin_username", "admin"),
        password=_config.get("auth", {}).get("admin_password", "admin"),
    )
    app.state.auth_service = auth_service
    logger.info("Auth service initialized (session_hours=%.1f)", session_hours)

    # Initialize modules (lazy load on first use to speed up startup)
    logger.info("API ready - modules will be initialized on first use")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _inspector
    if _inspector is not None:
        _inspector.close()
    logger.info("API shutdown complete")


def get_inspector():
    """Lazy-load the inspection module."""
    global _inspector, _config
    if _inspector is None:
        from inspection.visible import VisibleInspection
        _inspector = VisibleInspection(_config.get("inspection", {}))
        logger.info("VisibleInspection initialized")
    return _inspector


def get_teacher():
    """Lazy-load the teaching module."""
    global _teacher, _config
    if _teacher is None:
        # Import with try/except to handle both direct run and uvicorn contexts
        try:
            from src.teaching.tube_teacher import TubeTeacher
        except ImportError:
            from teaching.tube_teacher import TubeTeacher

        insp_cfg = _config.get("inspection", {})
        teach_cfg = _config.get("teaching", {})
        tube_cfg = insp_cfg.get("tube_pattern", {})
        _teacher = TubeTeacher(
            yolo_weights=insp_cfg.get("weights", {}).get("visible", "weights/visible_yolo.pt"),
            yolo_conf=insp_cfg.get("yolo_conf", 0.6),
            template_dir=teach_cfg.get("template_dir", tube_cfg.get("template_dir", "templates/tube")),
            bilateral_d=tube_cfg.get("bilateral_d", 9),
            bilateral_sigma_color=tube_cfg.get("bilateral_sigma_color", 75),
            bilateral_sigma_space=tube_cfg.get("bilateral_sigma_space", 75),
            device=teach_cfg.get("device", "auto"),
        )
        logger.info(
            "TubeTeacher initialized",
            extra={"template_dir": teach_cfg.get("template_dir", tube_cfg.get("template_dir"))}
        )
    return _teacher


def get_recipe_store():
    """Lazy-load the recipe store."""
    global _recipe_store, _config
    if _recipe_store is None:
        try:
            from src.inspection.recipe_store import RecipeStore
        except ImportError:
            from inspection.recipe_store import RecipeStore

        insp_cfg = _config.get("inspection", {}) if _config else {}
        recipe_dir = insp_cfg.get("recipe_dir", "data/recipes")
        _recipe_store = RecipeStore(recipe_dir)
        logger.info("RecipeStore initialized", extra={"recipe_dir": recipe_dir})
    return _recipe_store


# ============================================================================
# Tube Pattern Teaching - /tube + /retrain_all
# ============================================================================

# Project root for resolving relative paths (cv-code-0.5 → sieger_v2.2)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _resolve_folder(raw_path: str) -> Path:
    """Resolve a folder path, trying project root if not found directly."""
    folder = Path(raw_path)
    if folder.exists():
        return folder
    alt = _PROJECT_ROOT / folder.relative_to("/") if folder.is_absolute() else _PROJECT_ROOT / folder
    if alt.exists():
        logger.info(f"Resolved folder: {folder} → {alt}")
        return alt
    raise ValueError(f"Folder not found: {folder} (also tried {alt})")


def _teach_folder(folder: Path, teacher=None, result_folder: str = "") -> TubeTeachResponse:
    """Core teaching logic shared by /tube and /retrain_all.

    Reads master.json, loads images, runs teaching pipeline,
    updates recipe, runs validation on training images, and saves
    scored images to result_folder/good/ and result_folder/bad/.

    Raises:
        ValueError: Missing master.json, materialid, or insufficient images.
    """
    if teacher is None:
        teacher = get_teacher()

    # --- Read master.json ---
    master_json_path = folder / "master.json"
    if not master_json_path.exists():
        raise ValueError(f"master.json not found in {folder}")

    with open(master_json_path) as f:
        master_raw = json.load(f)
    master_data = master_raw[0] if isinstance(master_raw, list) and master_raw else master_raw

    material_id = str(
        master_data.get("materialid")
        or master_data.get("material_id")
        or ""
    )

    master_name = (
        master_data.get("masterid")
        or master_data.get("master_name")
        or master_data.get("master_id")
        or ""
    )

    if not master_name and not material_id:
        raise ValueError("master.json missing both 'masterid' and 'materialid' fields")

    cone_dia = float(master_data.get("conedia", 0) or master_data.get("cone_dia", 0) or 0)
    tube_dia = float(master_data.get("tubedia", 0) or master_data.get("tube_dia", 0) or 0)
    cone_tol = float(master_data.get("conetol", 0) or master_data.get("cone_tolerance", 0) or 0)
    tube_tol = float(master_data.get("tubetol", 0) or master_data.get("tube_tolerance", 0) or 0)

    # Teaching only needs master_name (pattern name for .npz template)
    # material_id is optional — only needed for recipe mapping
    teach_class = master_name if master_name else material_id

    logger.info(
        "master.json loaded: material_id=%s master_name=%s cone_dia=%.1f±%.1f tube_dia=%.1f±%.1f",
        material_id or "(none)", teach_class, cone_dia, cone_tol, tube_dia, tube_tol,
        extra={"material_id": material_id, "master_name": teach_class},
    )

    # --- Load images ---
    image_paths = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    if len(image_paths) < 2 and (folder / "good").exists():
        image_paths = list((folder / "good").glob("*.png")) + list((folder / "good").glob("*.jpg"))
        logger.info(f"Found {len(image_paths)} images in good/ subfolder")
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_paths)}")

    frames = []
    for p in sorted(image_paths):
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    if len(frames) < 2:
        raise ValueError("Could not load enough valid images")

    # --- Run teaching ---
    event_logger.log_teaching_started(material_id, len(image_paths))

    start_time = time.time()
    with PerformanceLogger("tube_teaching", logger, warn_threshold_ms=30000) as perf:
        result = teacher.teach(
            frames=frames,
            material_id=teach_class,
            save_crops_dir=str(folder),
        )
        perf.add_metric("material_id", material_id)
        perf.add_metric("teach_class", teach_class)
        perf.add_metric("n_frames", result["n_frames"])
        perf.add_metric("n_tubes_detected", result["n_tubes_detected"])

    duration_ms = (time.time() - start_time) * 1000

    # --- Save recipe (JSON file) — only if material_id is provided ---
    if material_id:
        get_recipe_store().upsert_recipe(
            material_id=material_id,
            master_name=master_name,
            cone_dia=cone_dia,
            tube_dia=tube_dia,
            cone_tol=cone_tol,
            tube_tol=tube_tol,
        )
        logger.info(
            "Recipe saved: id=%s master=%s cone_dia=%.1f±%.1f tube_dia=%.1f±%.1f",
            material_id, master_name, cone_dia, cone_tol, tube_dia, tube_tol,
        )
    else:
        logger.info(
            "No material_id in master.json — template saved, recipe skipped (create recipe separately)"
        )

    event_logger.log_teaching_completed(
        material_id=material_id,
        n_images=result["n_frames"],
        n_tubes_detected=result["n_tubes_detected"],
        duration_ms=duration_ms,
    )

    # --- Validation: run matcher on training images, save scored results ---
    if result_folder:
        try:
            # Reload templates so the just-taught one is available
            teacher.matcher.load_all_references()

            # Resolve path — frontend may send /sieger_data/... which needs absolute path resolution
            res_path = Path(result_folder)
            if not res_path.is_absolute() or not res_path.exists():
                alt = _PROJECT_ROOT / res_path.relative_to("/") if res_path.is_absolute() else _PROJECT_ROOT / res_path
                res_path = alt
            good_dir = res_path / "good"
            bad_dir = res_path / "bad"
            good_dir.mkdir(parents=True, exist_ok=True)
            bad_dir.mkdir(parents=True, exist_ok=True)

            sorted_paths = sorted(image_paths)
            for p, frame in zip(sorted_paths, frames):
                img_name = p.name

                # YOLO detect + extract tube crop
                detections = teacher.detector.detect(frame)
                tube_det = teacher.detector.get_detection_by_class(detections, "yarn_tube")
                if tube_det is None:
                    _overlay_tube_score(frame, "NO TUBE", scores={})
                    cv2.imwrite(str(bad_dir / img_name), frame)
                    continue

                tube_crop = teacher.detector.extract_annular_roi(
                    frame, tube_det, inner_ratio=teacher.matcher.inner_ratio,
                )
                match_result = teacher.matcher.verify(tube_crop, teach_class)

                label = "GOOD" if match_result.color_match else "BAD"
                scores = {
                    "bhatt": match_result.color_distance,
                    "combined": match_result.combined_distance,
                    "fft": match_result.fft_distance,
                }
                _overlay_tube_score(frame, label, scores)

                if match_result.color_match:
                    cv2.imwrite(str(good_dir / img_name), frame)
                else:
                    cv2.imwrite(str(bad_dir / img_name), frame)

            logger.info("Validation results saved to %s", result_folder)
        except Exception as e:
            logger.warning("Validation pass failed (non-fatal): %s", e)

    return TubeTeachResponse(
        status="success",
        training_id=material_id or teach_class,
        n_images=result["n_tubes_detected"],
        template_path=result["template_path"],
        result_folder=result_folder,
        color_threshold=result.get("color_threshold"),
        message=(
            f"Taught '{teach_class}' from {result['n_tubes_detected']}/{result['n_frames']} images "
            f"(Color NN + FFT NN, threshold={result.get('color_threshold', 0):.4f})"
        ),
    )


def _overlay_tube_score(image: np.ndarray, label: str, scores: dict) -> None:
    """Draw tube matching scores on the image (top-left corner)."""
    h, w = image.shape[:2]
    font_scale = max(0.6, min(w, h) / 1000.0)
    thickness = max(1, int(font_scale * 2))
    color = (0, 200, 0) if label == "GOOD" else (0, 0, 220)

    if scores:
        text = f"{label}  bhatt:{scores.get('bhatt', 0):.3f}  combined:{scores.get('combined', 0):.3f}"
    else:
        text = label

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    margin = int(10 * font_scale)
    cv2.rectangle(image, (0, 0), (tw + 2 * margin, th + 2 * margin + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (margin, th + margin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


@app.post("/tube", response_model=TubeTeachResponse, tags=["Teaching"])
async def teach_tube(request: TubeTeachRequest):
    """
    Teach tube pattern from images in a folder.

    Reads **master.json** from the folder to get:
    - `materialid`: PLC material number (required)
    - `masterid`: tube pattern class name (used as .npz template name)
    - `cone_dia`: reference cone diameter in mm
    - `tube_dia`: reference tube diameter in mm

    - **folder/image_folder**: Path to folder containing images + master.json
    """
    try:
        folder = _resolve_folder(request.image_folder)
        result = _teach_folder(folder, result_folder=request.result_folder)
        # Teaching requires inspection IDLE — templates loaded on next start
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Tube teaching failed", extra={"folder": request.image_folder})
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Retrain All Materials - /retrain_all
# ============================================================================

@app.post("/retrain_all", response_model=RetrainAllResponse, tags=["Teaching"])
async def retrain_all():
    """
    Retrain all tube patterns from sieger_data/teaching/VL/ folders.

    Scans for all folders containing master.json and re-runs the full
    teaching pipeline (same code path as /tube) for each.
    Useful after algorithm changes (e.g. crop geometry, feature extraction).
    """
    master_dir = Path(_config.get("data_root", "/home/msiegerips/sieger_data")) / "teaching" / "VL"

    if not master_dir.exists():
        raise HTTPException(status_code=400, detail=f"Teaching directory not found: {master_dir}")

    folders = sorted([p.parent for p in master_dir.glob("*/master.json")])
    if not folders:
        raise HTTPException(status_code=400, detail="No master.json files found in sieger_data/teaching/VL/")

    logger.info(f"Retrain all: found {len(folders)} materials in {master_dir}")

    teacher = get_teacher()
    results = []
    success_count = 0
    fail_count = 0

    for folder in folders:
        try:
            resp = _teach_folder(folder, teacher=teacher)
            results.append(RetrainResult(
                material_id=resp.material_id,
                master_name=folder.name,
                folder=str(folder),
                status="success",
                n_images=resp.n_images,
                message=resp.message,
            ))
            success_count += 1
        except Exception as e:
            results.append(RetrainResult(
                material_id=folder.name,
                master_name=folder.name,
                folder=str(folder),
                status="failed",
                message=str(e),
            ))
            fail_count += 1
            logger.exception(f"Retrain failed for {folder.name}")

    # Reload all templates so matcher picks up new .npz files
    teacher.matcher.load_all_references()

    return RetrainAllResponse(
        status="success" if fail_count == 0 else "partial",
        total=len(folders),
        success=success_count,
        failed=fail_count,
        results=results,
        message=f"Retrained {success_count}/{len(folders)} materials",
    )


# ============================================================================
# Stain Detection (Teaching + Evaluation) - /stain
# ============================================================================

@app.post("/stain", response_model=StainDetectResponse, tags=["Stain"])
async def stain_teach_or_detect(request: StainDetectRequest):
    """Stain detection: teach or detect.

    mode="teach": Train Gabor PCA model on good images from a folder.
    Fits the model, calibrates threshold (mean + k_sigma * std), saves
    model + calibration JSON, reloads the detector, and returns per-image
    scores as evaluation results.

    mode="detect" (default): Run stain detection on images using the
    existing trained model.

    Parameters:
        id/material_id: Material identifier
        folder/image_folder: Path to folder with images
        mode: "teach" or "detect"
        k_sigma: Threshold sigma multiplier (teach mode, default 3.0)
    """
    try:
        inspector = get_inspector()

        if inspector.stain_detector is None:
            return StainDetectResponse(
                status="not_configured",
                training_id=request.training_id,
                n_images=0,
                message="Stain detector not initialized. Check model path in config.",
            )

        # Collect images
        images_to_process = []

        if request.image_folder:
            folder = Path(request.image_folder)
            if not folder.exists():
                raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
            for p in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
                img = cv2.imread(str(p))
                if img is not None:
                    images_to_process.append((p.name, img))

        elif request.image_path:
            img = cv2.imread(request.image_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not load image")
            images_to_process.append((Path(request.image_path).name, img))

        elif request.image_base64:
            img_bytes = base64.b64decode(request.image_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            images_to_process.append(("uploaded_image", img))

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide image_folder, image_path, or image_base64",
            )

        if not images_to_process:
            raise HTTPException(status_code=400, detail="No valid images found")

        try:
            from src.inspection.polar_unwarp import find_geometry
            from src.inspection.stain_detector import _unwarp_with_mask
        except ImportError:
            from inspection.polar_unwarp import find_geometry
            from inspection.stain_detector import _unwarp_with_mask

        # ---- TEACH MODE ----
        if request.mode == "teach":
            # Accepts multiple material_id folders via "folders" field,
            # or a single folder via "folder". Each folder has VL/good/ and VL/bad/.
            # All good images are combined for training, all bad for evaluation.

            # Collect all folder paths
            folder_paths = []
            if request.image_folders:
                folder_paths = [Path(f) for f in request.image_folders]
            elif request.image_folder:
                folder_paths = [Path(request.image_folder)]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Provide 'folder' or 'folders' for teach mode",
                )

            # Scan all folders for VL/good and VL/bad
            good_paths = []
            bad_paths = []
            missing_folders = []

            for fp in folder_paths:
                if not fp.exists():
                    missing_folders.append(str(fp))
                    continue

                good_dir = fp / "VL" / "good"
                bad_dir = fp / "VL" / "bad"

                if good_dir.exists():
                    good_paths.extend(
                        sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
                    )
                if bad_dir.exists():
                    bad_paths.extend(
                        sorted(bad_dir.glob("*.png")) + sorted(bad_dir.glob("*.jpg"))
                    )

            if missing_folders:
                logger.warning("Folders not found: %s", missing_folders)

            if len(good_paths) < 5:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Need at least 5 good images for training, found {len(good_paths)} "
                        f"across {len(folder_paths)} folder(s). "
                        f"Expected structure: {{folder}}/VL/good/*.png"
                    ),
                )

            # Load good images
            good_images = []
            for p in good_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    good_images.append((p.name, img))

            # Load bad images (for evaluation only)
            bad_images = []
            for p in bad_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    bad_images.append((p.name, img))

            logger.info(
                "Stain teaching started: %d good + %d bad images from %d folder(s), k_sigma=%.1f",
                len(good_images), len(bad_images), len(folder_paths), request.k_sigma,
            )

            # Step 1: YOLO + polar unwrap good images for training
            strips = []
            masks = []
            skipped = []

            for img_name, frame in good_images:
                detections = inspector.detector.detect(frame)
                cone_det = inspector.detector.get_detection_by_class(detections, "yarn_cone")
                tube_det = inspector.detector.get_detection_by_class(detections, "yarn_tube")

                if cone_det is None or tube_det is None:
                    skipped.append(img_name)
                    continue

                cone_crop = inspector.detector.extract_roi(frame, cone_det)
                center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)
                strip, mask = _unwarp_with_mask(cone_crop, center, inner_r, outer_r)
                strips.append(strip)
                masks.append(mask)

            if len(strips) < 5:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Need at least 5 valid good images for training, got {len(strips)} "
                        f"({len(skipped)} skipped: no cone/tube detected)"
                    ),
                )

            # Step 2: Build annular (donut) crops and write to a temp MVTec-like dataset dir
            import shutil
            import tempfile
            from datetime import datetime, timezone

            tmp_root = Path(tempfile.mkdtemp(prefix="patchcore_teach_"))
            try:
                train_good_dir = tmp_root / "train" / "good"
                test_good_dir = tmp_root / "test" / "good"
                train_good_dir.mkdir(parents=True)
                test_good_dir.mkdir(parents=True)

                # Apply annular mask to each YOLO-cropped cone (same as inference path)
                for idx, (cone_crop, mask) in enumerate(zip(strips, masks)):
                    donut = cone_crop.copy()
                    donut[mask == 0] = 0
                    # 80/20 train/test split
                    if idx < int(len(strips) * 0.8):
                        out_path = train_good_dir / f"{idx:04d}.png"
                    else:
                        out_path = test_good_dir / f"{idx:04d}.png"
                    cv2.imwrite(str(out_path), donut)

                logger.info(
                    "Stain teach: wrote %d donut crops to %s (%d train / %d test)",
                    len(strips), tmp_root,
                    len(list(train_good_dir.glob("*.png"))),
                    len(list(test_good_dir.glob("*.png"))),
                )

                # Step 3: Train PatchCore via anomalib
                import os
                import torch
                os.environ["TRUST_REMOTE_CODE"] = "1"
                torch.set_float32_matmul_precision("medium")

                # Patch anomalib for pandas 3.0 compatibility before importing
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

                    samples["label_index"] = int(LabelName.NORMAL)
                    is_abnormal = samples.label == ABNORMAL
                    samples.loc[is_abnormal, "label_index"] = int(LabelName.ABNORMAL)
                    samples["label_index"] = samples["label_index"].astype("Int64")
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
                    samples.attrs["task"] = "classification"
                    if split:
                        samples = samples[samples.split == str(split)]
                        samples = samples.reset_index(drop=True)
                    return samples

                folder_mod.make_folder_dataset = _patched_make_folder_dataset

                from anomalib.data import Folder
                from anomalib.data.utils import TestSplitMode, ValSplitMode
                from anomalib.deploy import ExportType
                from anomalib.engine import Engine
                from anomalib.models import Patchcore

                datamodule = Folder(
                    name="cone_surface",
                    root=str(tmp_root),
                    normal_dir="train/good",
                    normal_test_dir="test/good",
                    train_batch_size=32,
                    eval_batch_size=32,
                    test_split_mode=TestSplitMode.FROM_DIR,
                    val_split_mode=ValSplitMode.SAME_AS_TEST,
                )

                model = Patchcore(
                    backbone="wide_resnet50_2",
                    layers=("layer2", "layer3"),
                    pre_trained=True,
                    coreset_sampling_ratio=0.1,
                    num_neighbors=9,
                )

                train_output_dir = tmp_root / "results"
                engine = Engine(default_root_dir=str(train_output_dir))

                logger.info("Stain teach: starting PatchCore training (%d good images)", len(strips))
                engine.fit(model=model, datamodule=datamodule)

                # Step 4: Export and copy to production model path
                export_root = train_output_dir / "Patchcore" / "cone_surface" / "exported"
                engine.export(
                    model=model,
                    export_type=ExportType.TORCH,
                    export_root=str(export_root / "torch"),
                )

                exported_model = export_root / "torch" / "weights" / "torch" / "model.pt"
                prod_model_path = Path("models/patchcore/weights/torch/model.pt")
                prod_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(exported_model), str(prod_model_path))

                logger.info("Stain model saved to %s", prod_model_path)

                # Save calibration record
                cal = {
                    "setup_id": request.training_id,
                    "model": "patchcore",
                    "backbone": "wide_resnet50_2",
                    "n_images": len(strips),
                    "date": datetime.now(timezone.utc).isoformat(),
                }
                cal_path = prod_model_path.parent / f"calibration_{request.training_id}.json"
                for old_cal in prod_model_path.parent.glob("calibration_*.json"):
                    old_cal.unlink()
                cal_path.write_text(json.dumps(cal, indent=2))

            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

            # Step 5: Reload detector with new model
            inspector.stain_detector._load_model(str(prod_model_path))

            # Step 5: Evaluate on both good + bad images
            results = []

            def _eval_images(image_list, label):
                """Run inference on a list of (name, frame) and return results."""
                eval_results = []
                for img_name, frame in image_list:
                    detections = inspector.detector.detect(frame)
                    cone_det = inspector.detector.get_detection_by_class(detections, "yarn_cone")
                    tube_det = inspector.detector.get_detection_by_class(detections, "yarn_tube")

                    if cone_det is None or tube_det is None:
                        eval_results.append(StainDetectResult(
                            image_name=f"[{label}] {img_name}",
                            has_stain=False,
                            anomaly_score=-1.0,
                            heatmap_base64=None,
                        ))
                        continue

                    cone_crop = inspector.detector.extract_roi(frame, cone_det)
                    center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)
                    sr = inspector.stain_detector.detect(
                        cone_crop, center=center, inner_r=inner_r, outer_r=outer_r,
                    )

                    eval_results.append(StainDetectResult(
                        image_name=f"[{label}] {img_name}",
                        has_stain=sr.has_stain,
                        anomaly_score=round(sr.anomaly_score, 4),
                        heatmap_base64=None,
                    ))
                return eval_results

            results.extend(_eval_images(good_images, "good"))
            results.extend(_eval_images(bad_images, "bad"))

            good_count = sum(
                1 for r in results
                if r.image_name.startswith("[good]") and not r.has_stain and r.anomaly_score >= 0
            )
            bad_detected = sum(
                1 for r in results
                if r.image_name.startswith("[bad]") and r.has_stain
            )
            stain_count = sum(1 for r in results if r.has_stain)

            return StainDetectResponse(
                status="success",
                training_id=request.training_id,
                n_images=len(results),
                results=results,
                good_count=good_count,
                stain_count=stain_count,
                threshold=None,
                mean_score=None,
                std_score=None,
                model_path=str(prod_model_path),
                message=(
                    f"PatchCore model trained on {len(strips)} good images ({len(skipped)} skipped). "
                    f"Model saved to {prod_model_path}. "
                    f"Evaluation: {good_count}/{len(good_images)} good correct, "
                    f"{bad_detected}/{len(bad_images)} bad detected."
                ),
            )

        # ---- DETECT MODE (default) ----
        results = []
        good_count = 0
        stain_count = 0

        for img_name, frame in images_to_process:
            detections = inspector.detector.detect(frame)
            cone_det = inspector.detector.get_detection_by_class(detections, "yarn_cone")
            tube_det = inspector.detector.get_detection_by_class(detections, "yarn_tube")

            if cone_det is None or tube_det is None:
                results.append(StainDetectResult(
                    image_name=img_name,
                    has_stain=False,
                    anomaly_score=-1.0,
                    heatmap_base64=None,
                ))
                continue

            cone_crop = inspector.detector.extract_roi(frame, cone_det)
            center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)

            stain_result = inspector.stain_detector.detect(
                cone_crop, center=center, inner_r=inner_r, outer_r=outer_r,
            )

            heatmap_b64 = None
            if stain_result.heatmap is not None:
                hm_norm = stain_result.heatmap.copy()
                if hm_norm.max() > 0:
                    hm_norm = hm_norm / hm_norm.max()
                hm_uint8 = (np.clip(hm_norm, 0, 1) * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                _, buffer = cv2.imencode(".png", hm_color)
                heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

            results.append(StainDetectResult(
                image_name=img_name,
                has_stain=stain_result.has_stain,
                anomaly_score=round(stain_result.anomaly_score, 4),
                heatmap_base64=heatmap_b64,
            ))

            if stain_result.has_stain:
                stain_count += 1
            else:
                good_count += 1

        return StainDetectResponse(
            status="success",
            training_id=request.training_id,
            n_images=len(results),
            results=results,
            good_count=good_count,
            stain_count=stain_count,
            threshold=round(inspector.stain_detector.threshold, 2),
            message=f"Processed {len(results)} images: {good_count} good, {stain_count} with stain",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Stain teach/detect failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Extract (Cone Dimensions) - /extract
# ============================================================================

@app.post("/extract", response_model=ExtractResponse, tags=["Teaching"])
async def extract_cone(request: ExtractRequest):
    """
    Extract cone dimensions from images for calibration.

    Runs YOLO detection to get cone and tube bounding boxes,
    then calculates dimensions using pixels_per_mm calibration.

    - **id/material_id**: Material identifier
    - **folder/image_folder**: Path to folder containing cone images
    - **scale**: Pixels per mm (calibration constant)
    """
    try:
        inspector = get_inspector()

        folder = Path(request.image_folder)
        if not folder.exists():
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

        image_paths = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if not image_paths:
            raise HTTPException(status_code=400, detail="No images found in folder")

        # Process first image for dimension extraction
        img = cv2.imread(str(image_paths[0]))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not load image")

        # Run detection
        detections = inspector.detector.detect(img)
        cone_det = inspector.detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = inspector.detector.get_detection_by_class(detections, "yarn_tube")

        cone_dia_mm = None
        tube_dia_mm = None
        pixels_per_mm = request.scale or _config.get("inspection", {}).get("pixels_per_mm", 5.0)

        if cone_det:
            x1, y1, x2, y2 = cone_det.bbox
            cone_dia_mm = round(min(x2 - x1, y2 - y1) / pixels_per_mm, 1)

        if tube_det:
            x1, y1, x2, y2 = tube_det.bbox
            tube_dia_mm = round(min(x2 - x1, y2 - y1) / pixels_per_mm, 1)

        return ExtractResponse(
            status="success",
            material_id=request.material_id,
            cone_diameter_mm=cone_dia_mm,
            tube_diameter_mm=tube_dia_mm,
            pixels_per_mm=pixels_per_mm,
            message=f"Extracted from {len(image_paths)} images",
        )

    except Exception as e:
        logger.exception("Extract failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Color Detection - /color_detection
# ============================================================================

@app.post("/color_detection", response_model=ColorDetectResponse, tags=["Teaching"])
async def color_detection(request: ColorDetectRequest):
    """
    Color detection and clustering for a material.

    Note: Color is now part of tube pattern teaching (/tube endpoint).
    This endpoint is kept for backward compatibility.
    """
    return ColorDetectResponse(
        status="deprecated",
        message="Color detection is now integrated into /tube endpoint",
    )


# ============================================================================
# Delete Master - /delete_master
# ============================================================================

@app.post("/delete_master", response_model=DeleteMasterResponse, tags=["Teaching"])
async def delete_master(request: DeleteMasterRequest):
    """
    Delete teaching data for a material.

    Removes .npz template file and database record.

    - **mat_id/material_id**: Material identifier to delete
    """
    try:
        teacher = get_teacher()
        deleted = teacher.delete_reference(request.material_id)

        if deleted:
            return DeleteMasterResponse(
                status="success",
                message=f"Deleted teaching data for {request.material_id}",
            )
        else:
            return DeleteMasterResponse(
                status="not_found",
                message=f"No teaching data found for {request.material_id}",
            )

    except Exception as e:
        logger.exception("Delete master failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Get Teaching Data - /get_teaching_data
# ============================================================================

@app.post("/get_teaching_data", response_model=GetTeachingDataResponse, tags=["Teaching"])
async def get_teaching_data():
    """
    Get list of all taught materials.

    Returns material IDs with metadata (n_images, created_at, etc.)
    """
    try:
        teacher = get_teacher()
        refs = teacher.list_references()

        data = [
            TeachingDataItem(
                mat_id=r["material_id"],
                master=r["material_id"],  # For backward compatibility
                n_images=r.get("n_images"),
                created_at=r.get("created_at"),
            )
            for r in refs
        ]

        return GetTeachingDataResponse(status="success", data=data)

    except Exception as e:
        logger.exception("Get teaching data failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recipe CRUD - /recipes + /masters
# ============================================================================

@app.get("/recipes", response_model=RecipeListResponse, tags=["Recipes"])
async def list_recipes():
    """List all material recipes (JSON files in data/recipes/)."""
    try:
        store = get_recipe_store()
        recipes = store.list_recipes()
        return RecipeListResponse(status="success", recipes=recipes)
    except Exception as e:
        logger.exception("List recipes failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recipes", response_model=RecipeResponse, tags=["Recipes"])
async def create_or_update_recipe(request: RecipeRequest):
    """Create or update a material recipe.

    Maps a PLC material_id to a master_name + dimensions + tolerances.
    """
    try:
        mat_id = request.get_material_id()
        master = request.get_master_name()
        if not mat_id:
            raise HTTPException(status_code=400, detail="material_id or materialid is required")
        if not master:
            raise HTTPException(status_code=400, detail="master_name or masterid is required")

        store = get_recipe_store()
        recipe = store.upsert_recipe(
            material_id=mat_id,
            master_name=master,
            cone_dia=request.get_cone_dia(),
            tube_dia=request.get_tube_dia(),
            cone_tol=request.get_cone_tol(),
            tube_tol=request.get_tube_tol(),
        )
        logger.info(
            "Recipe upserted: id=%s master=%s cone=%.1f±%.1f tube=%.1f±%.1f",
            mat_id, master, request.get_cone_dia(), request.get_cone_tol(),
            request.get_tube_dia(), request.get_tube_tol(),
        )
        return RecipeResponse(status="success", recipe=recipe)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Create/update recipe failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/recipes/{material_id}", response_model=RecipeResponse, tags=["Recipes"])
async def delete_recipe(material_id: str):
    """Delete a material recipe."""
    try:
        store = get_recipe_store()
        deleted = store.delete_recipe(material_id)
        if deleted:
            return RecipeResponse(status="success", message=f"Recipe '{material_id}' deleted")
        else:
            raise HTTPException(status_code=404, detail=f"Recipe '{material_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete recipe failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/masters", response_model=MasterListResponse, tags=["Teaching"])
async def list_masters():
    """List all taught master_names from .npz files on disk."""
    try:
        teacher = get_teacher()
        refs = teacher.list_references()
        master_names = sorted({r["material_id"] for r in refs})
        return MasterListResponse(status="success", masters=master_names)
    except Exception as e:
        logger.exception("List masters failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tube OCR - /tube_ocr (Placeholder)
# ============================================================================

@app.post("/tube_ocr", response_model=TubeOCRResponse, tags=["Teaching"])
async def tube_ocr(request: TubeOCRRequest):
    """
    Store OCR data for a material.

    Placeholder endpoint - will be implemented later.
    """
    # TODO: Implement OCR data storage
    return TubeOCRResponse(
        status="not_implemented",
        message="OCR endpoint placeholder - will be implemented later",
    )


# ============================================================================
# Get Tube Image - /get_tube_img
# ============================================================================

@app.post("/get_tube_img", tags=["Teaching"])
async def get_tube_img(material_id: str = None, image_path: str = None):
    """
    Get processed tube image with annotations.

    Runs YOLO detection and returns annotated tube ROI.
    """
    try:
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path required")

        inspector = get_inspector()

        img = cv2.imread(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not load image")

        detections = inspector.detector.detect(img)
        tube_det = inspector.detector.get_detection_by_class(detections, "yarn_tube")

        if tube_det is None:
            raise HTTPException(status_code=404, detail="No tube detected in image")

        tube_crop = inspector.detector.extract_roi(img, tube_det)

        # Encode as base64
        _, buffer = cv2.imencode(".png", tube_crop)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "status": "success",
            "image_base64": img_base64,
            "bbox": tube_det.bbox,
            "confidence": tube_det.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Get tube img failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Runtime Inspection - /inspect
# ============================================================================

@app.post("/inspect", response_model=InspectResponse, tags=["Inspection"])
async def inspect(request: InspectRequest):
    """
    Run full inspection on a single frame.

    Performs YOLO detection, dimension check, stain detection, and tube pattern verification.

    - **material_id**: Expected material ID for verification
    - **image_path**: Path to image file (or)
    - **image_base64**: Base64 encoded image

    Returns result_code: 1=Good, 2=Defect, 3=Error
    """
    try:
        inspector = get_inspector()

        # Load image
        if request.image_path:
            frame = cv2.imread(request.image_path)
            if frame is None:
                raise HTTPException(status_code=400, detail="Could not load image")
        elif request.image_base64:
            img_bytes = base64.b64decode(request.image_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
        else:
            raise HTTPException(status_code=400, detail="image_path or image_base64 required")

        # Run inspection
        result, annotated = inspector.process_frame_with_visualization(frame, request.material_id)

        # Encode annotated image
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = base64.b64encode(buffer).decode("utf-8")

        # Extract dimension info
        cone_dia = None
        tube_dia = None
        dims_ok = None
        if result.dimension_result:
            cone_dia = result.dimension_result.measured.cone_diameter_mm
            tube_dia = result.dimension_result.measured.tube_diameter_mm
            dims_ok = result.dimension_result.all_match

        # Extract stain info
        stain_detected = None
        if result.stain_result:
            stain_detected = result.stain_result.has_stain

        # Extract tube pattern info
        tube_ok = None
        if result.tube_pattern_result:
            tube_ok = result.tube_pattern_result.passed

        return InspectResponse(
            status="success",
            result_code=result.result_code,
            passed=result.passed,
            material_id=request.material_id,
            dimensions_ok=dims_ok,
            stain_detected=stain_detected,
            tube_pattern_ok=tube_ok,
            cone_diameter_mm=cone_dia,
            tube_diameter_mm=tube_dia,
            annotated_image_base64=annotated_base64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inspection failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status():
    """Get API status and health information."""
    global _inspector, _start_time

    templates_loaded = 0
    if _inspector and hasattr(_inspector, 'tube_matcher') and _inspector.tube_matcher:
        templates_loaded = len(_inspector.tube_matcher._templates)

    return StatusResponse(
        status="running",
        yolo_loaded=_inspector is not None,
        templates_loaded=templates_loaded,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/restart", response_model=RestartResponse, tags=["System"])
async def restart(background_tasks: BackgroundTasks):
    """
    Restart the inspection service.

    Reinitializes all modules (YOLO, templates, etc.)
    """
    global _inspector, _teacher

    def do_restart():
        global _inspector, _teacher, _recipe_store
        if _inspector:
            _inspector.close()
        _inspector = None
        _teacher = None
        _recipe_store = None
        logger.info("Service restarted - modules will reinitialize on next use")

    background_tasks.add_task(do_restart)

    return RestartResponse(
        status="success",
        message="Restart initiated",
    )


@app.post("/shutdown", response_model=ShutdownResponse, tags=["System"])
async def shutdown(background_tasks: BackgroundTasks):
    """
    Shutdown the inspection service.

    Gracefully stops the API server.
    """
    def do_shutdown():
        time.sleep(1)  # Give time for response
        os.kill(os.getpid(), signal.SIGTERM)

    background_tasks.add_task(do_shutdown)

    return ShutdownResponse(
        status="success",
        message="Shutdown initiated",
    )


# ============================================================================
# Health Check Endpoints
# ============================================================================

def _check_plc_connection() -> PLCHealthStatus:
    """Check PLC connection status."""
    try:
        plc_config = _config.get("plc", {}) if _config else {}
        host = plc_config.get("host", "192.168.2.1")
        port = plc_config.get("port", 502)

        # Try to connect to PLC
        from pyModbusTCP.client import ModbusClient
        client = ModbusClient(host=host, port=port, timeout=2.0)
        connected = client.open()
        if connected:
            client.close()

        return PLCHealthStatus(
            connected=connected,
            host=host,
            port=port,
            error=None if connected else "Connection failed",
        )
    except ImportError:
        return PLCHealthStatus(
            connected=False,
            host=plc_config.get("host"),
            port=plc_config.get("port"),
            error="pyModbusTCP not installed",
        )
    except Exception as e:
        return PLCHealthStatus(
            connected=False,
            host=plc_config.get("host") if _config else None,
            port=plc_config.get("port") if _config else None,
            error=str(e),
        )


def _check_camera_connection(cam_name: str, cam_config: dict) -> CameraHealthStatus:
    """Check a single camera connection status."""
    try:
        ip = cam_config.get("ip", "")
        exposure = cam_config.get("exposure", 0)

        # Try to ping the camera IP (simple connectivity check)
        import subprocess
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", ip],
            capture_output=True,
            timeout=2,
        )
        connected = result.returncode == 0

        return CameraHealthStatus(
            name=cam_name.lower(),
            connected=connected,
            ip=ip,
            exposure=exposure,
            error=None if connected else "Camera not reachable",
        )
    except Exception as e:
        return CameraHealthStatus(
            name=cam_name.lower(),
            connected=False,
            ip=cam_config.get("ip"),
            exposure=cam_config.get("exposure"),
            error=str(e),
        )


@app.get("/health", tags=["Health"])
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/health/system", response_model=SystemHealthResponse, tags=["Health"])
async def health_system():
    """
    Get comprehensive system health status.

    Checks PLC, all cameras, and ML models.
    Returns overall status: 'healthy', 'degraded', or 'unhealthy'.
    """
    global _inspector, _start_time, _config

    # Check PLC
    plc_status = _check_plc_connection()

    # Check cameras
    cameras_config = _config.get("cameras", {}) if _config else {}
    camera_statuses = []
    for cam_name, cam_cfg in cameras_config.items():
        cam_status = _check_camera_connection(cam_name, cam_cfg)
        camera_statuses.append(cam_status)

    # Check models
    yolo_loaded = _inspector is not None and hasattr(_inspector, 'detector') and _inspector.detector is not None
    patchcore_loaded = _inspector is not None and hasattr(_inspector, 'stain_detector') and _inspector.stain_detector is not None

    # Determine overall status
    all_cameras_ok = all(c.connected for c in camera_statuses) if camera_statuses else True
    if plc_status.connected and all_cameras_ok:
        overall_status = "healthy"
        message = "All systems operational"
    elif plc_status.connected or all_cameras_ok:
        overall_status = "degraded"
        issues = []
        if not plc_status.connected:
            issues.append("PLC disconnected")
        disconnected_cams = [c.name for c in camera_statuses if not c.connected]
        if disconnected_cams:
            issues.append(f"Cameras offline: {', '.join(disconnected_cams)}")
        message = "; ".join(issues)
    else:
        overall_status = "unhealthy"
        message = "PLC and cameras disconnected"

    return SystemHealthResponse(
        status=overall_status,
        plc=plc_status,
        cameras=camera_statuses,
        models_loaded=yolo_loaded,
        yolo_loaded=yolo_loaded,
        patchcore_loaded=patchcore_loaded,
        uptime_seconds=round(time.time() - _start_time, 1),
        message=message,
    )


@app.get("/health/plc", response_model=PLCHealthResponse, tags=["Health"])
async def health_plc():
    """
    Check PLC connection status.

    Tests Modbus TCP connection to the configured PLC.
    """
    plc_status = _check_plc_connection()

    return PLCHealthResponse(
        status="connected" if plc_status.connected else "disconnected",
        plc=plc_status,
        message="PLC is reachable" if plc_status.connected else plc_status.error,
    )


@app.get("/health/cameras", response_model=CameraHealthResponse, tags=["Health"])
async def health_cameras():
    """
    Check all camera connections.

    Pings each configured camera IP to verify network connectivity.
    """
    global _config

    cameras_config = _config.get("cameras", {}) if _config else {}
    camera_statuses = []

    for cam_name, cam_cfg in cameras_config.items():
        cam_status = _check_camera_connection(cam_name, cam_cfg)
        camera_statuses.append(cam_status)

    all_connected = all(c.connected for c in camera_statuses) if camera_statuses else False

    if not camera_statuses:
        message = "No cameras configured"
    elif all_connected:
        message = f"All {len(camera_statuses)} cameras connected"
    else:
        disconnected = [c.name for c in camera_statuses if not c.connected]
        message = f"Cameras offline: {', '.join(disconnected)}"

    return CameraHealthResponse(
        status="ok" if all_connected else "degraded",
        cameras=camera_statuses,
        all_connected=all_connected,
        message=message,
    )


@app.get("/health/camera/{camera_name}", response_model=CameraHealthStatus, tags=["Health"])
async def health_camera_single(camera_name: str):
    """
    Check a specific camera connection.

    Args:
        camera_name: Camera name ('VL', 'UV', or 'Tail')
    """
    global _config

    cameras_config = _config.get("cameras", {}) if _config else {}

    # Find camera (case-insensitive)
    cam_cfg = None
    matched_name = None
    for name, cfg in cameras_config.items():
        if name.lower() == camera_name.lower():
            cam_cfg = cfg
            matched_name = name
            break

    if cam_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_name}' not found. Available: {list(cameras_config.keys())}"
        )

    return _check_camera_connection(matched_name, cam_cfg)




# ============================================================================
# Capture & Teaching — /capture  /teaching
# ============================================================================

VALID_MODULES = {"tube", "stain", "uv", "tail", "dimension"}


def _emit_to_inspection(event: str, data: dict) -> dict:
    """Emit a socket.io event to the inspection service and return the response."""
    import socketio as sio_client
    inspection_port = _config.get("service", {}).get("port", 5004) if _config else 5004
    result = {}
    client = sio_client.SimpleClient()
    client.connect(f"http://localhost:{inspection_port}", wait_timeout=3)
    client.emit(event, data)
    ev = client.receive(timeout=3)
    if ev:
        result = ev[1] if isinstance(ev, list) and len(ev) > 1 else {}
    client.disconnect()
    return result


@app.get("/capture/status", tags=["Capture"])
async def capture_status():
    """Return current capture mode state from inspection service."""
    try:
        status = _emit_to_inspection("get_capture_status", {})
        return status
    except Exception as e:
        logger.warning("Could not get capture status from inspection service: %s", e)
        return {"active": False, "session_id": "", "module": "", "material_ids": []}


@app.get("/capture/sessions", tags=["Capture"])
async def capture_sessions(limit: int = 50, offset: int = 0):
    """List capture sessions (audit trail), newest first."""
    db = _get_db()
    try:
        rows = db.execute(
            """SELECT session_id, module, material_ids, started_at, stopped_at,
                      images_saved, stopped_by
               FROM capture_sessions
               ORDER BY started_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return {"sessions": [dict(r) for r in rows], "limit": limit, "offset": offset}
    finally:
        db.close()


@app.get("/capture/images", tags=["Capture"])
async def capture_images(
    session_id: str = None,
    module: str = None,
    material_id: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """List captured images with optional filters."""
    db = _get_db()
    try:
        clauses = []
        params = []
        if session_id:
            clauses.append("session_id = ?"); params.append(session_id)
        if module:
            clauses.append("module = ?"); params.append(module)
        if material_id:
            clauses.append("material_id = ?"); params.append(material_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = db.execute(
            f"""SELECT image_id, session_id, material_id, module, captured_at,
                       vl_path, uv_path, tail_path
                FROM captured_images
                {where}
                ORDER BY captured_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        total = db.execute(
            f"SELECT COUNT(*) FROM captured_images {where}", params
        ).fetchone()[0]
        return {"images": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}
    finally:
        db.close()


@app.post("/teaching/annotate", tags=["Teaching"])
async def teaching_annotate(request: Request):
    """Save good/bad/discard labels for captured images.

    Body: {annotations: [{image_id, module, label, annotated_by}]}

    Labels:
        good    — normal cone, used for training
        bad     — confirmed defect, used for validation (never trained on)
        discard — unusable image (blur/partial/wrong material), excluded

    Same image can have independent labels per module.
    """
    body = await request.json()
    annotations = body.get("annotations", [])
    if not annotations:
        raise HTTPException(status_code=400, detail="annotations list required")

    now = datetime.now(timezone.utc).isoformat()
    db = _get_db()
    try:
        for ann in annotations:
            image_id = ann.get("image_id", "")
            module = ann.get("module", "")
            label = ann.get("label", "")
            annotated_by = ann.get("annotated_by", "")

            if not image_id or not module or label not in ("good", "bad", "discard"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Each annotation needs image_id, module, and label in ('good','bad','discard'). Got: {ann}"
                )

            # Upsert — operator can change their mind
            db.execute(
                """INSERT INTO image_annotations (annotation_id, image_id, module, label, annotated_at, annotated_by)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(annotation_id) DO NOTHING""",
                (str(uuid.uuid4()), image_id, module, label, now, annotated_by),
            )
            # Also update by (image_id, module) uniqueness
            db.execute(
                """INSERT INTO image_annotations (annotation_id, image_id, module, label, annotated_at, annotated_by)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (str(uuid.uuid4()), image_id, module, label, now, annotated_by),
            )

        db.commit()
    finally:
        db.close()

    return {"annotated": len(annotations)}



@app.post("/teaching/tube", tags=["Teaching"])
async def teaching_tube(request: Request):
    """Train tube pattern model from good-labelled captured images.

    Body: {material_id: str, scope_key: str (optional, defaults to material_id)}

    Queries captured_images + image_annotations for good-labelled VL images
    for the given material_id, loads PNGs from disk, runs TubeTeacher.teach(),
    saves .npz template, writes teaching_session to SQLite, notifies inspection
    service to reload templates.

    Returns 409 if no good-labelled images found.
    Returns 409 if inspection is currently running (check /capture/status first).
    """
    import cv2 as _cv2
    import numpy as _np

    body = await request.json()
    material_id = body.get("material_id", "")
    scope_key = body.get("scope_key", material_id)
    annotated_by = body.get("annotated_by", "")

    if not material_id:
        raise HTTPException(status_code=400, detail="material_id required")

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    # Query good-labelled VL images for this material_id
    db = _get_db()
    try:
        rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'tube'
               WHERE ci.material_id = ?
                 AND ci.module = 'tube'
                 AND ia.label = 'good'
                 AND ci.vl_path IS NOT NULL
               ORDER BY ci.captured_at ASC""",
            (material_id,),
        ).fetchall()
    finally:
        db.close()

    if not rows:
        raise HTTPException(
            status_code=409,
            detail=f"No good-labelled tube images found for material_id='{material_id}'. "
                   "Capture images and annotate them as good first."
        )

    # Load PNG frames from disk
    frames = []
    missing = []
    for row in rows:
        img_path = data_root / row["vl_path"]
        frame = _cv2.imread(str(img_path))
        if frame is None:
            missing.append(str(img_path))
            continue
        frames.append(frame)

    if missing:
        logger.warning("teaching/tube: %d image files not found on disk: %s", len(missing), missing[:5])

    if len(frames) < 2:
        raise HTTPException(
            status_code=409,
            detail=f"Need at least 2 readable images, got {len(frames)} "
                   f"({len(missing)} missing from disk)"
        )

    # Run TubeTeacher
    teacher = get_teacher()
    teaching_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = teacher.teach(frames=frames, material_id=material_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("TubeTeacher.teach() failed for material_id=%s", material_id)
        raise HTTPException(status_code=500, detail=f"Teaching failed: {e}")

    completed_at = datetime.now(timezone.utc).isoformat()

    # Write teaching_session to SQLite
    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, completed_at,
                n_samples, model_path, threshold, extend_count, notes)
               VALUES (?, 'tube', ?, 'active', ?, ?, ?, ?, ?, 0, ?)""",
            (
                teaching_id, scope_key, started_at, completed_at,
                result["n_tubes_detected"],
                result["template_path"],
                result["color_threshold"],
                f"trained from {len(frames)} captured images",
            ),
        )
        # Supersede any previous active tube session for this scope_key
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='tube' AND scope_key=? AND status='active'
               AND teaching_id != ?""",
            (scope_key, teaching_id),
        )
        db.commit()
    finally:
        db.close()

    # Teaching requires inspection IDLE — new templates loaded on next inspection start

    logger.info(
        "Tube teaching complete: material=%s teaching_id=%s n_samples=%d threshold=%.4f",
        material_id, teaching_id, result["n_tubes_detected"], result["color_threshold"],
    )

    return {
        "teaching_id": teaching_id,
        "material_id": material_id,
        "scope_key": scope_key,
        "n_frames_input": len(frames),
        "n_tubes_detected": result["n_tubes_detected"],
        "color_threshold": result["color_threshold"],
        "template_path": result["template_path"],
        "completed_at": completed_at,
    }


@app.post("/teaching/tube/extend", tags=["Teaching"])
async def teaching_tube_extend(request: Request):
    """Extend existing tube pattern model with additional good-labelled images.

    Body: {material_id: str}

    Uses newly captured + annotated images that are not yet part of any
    teaching session. Capped at 3 extensions before full re-teach required.
    """
    import cv2 as _cv2

    body = await request.json()
    material_id = body.get("material_id", "")
    scope_key = body.get("scope_key", material_id)

    if not material_id:
        raise HTTPException(status_code=400, detail="material_id required")

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    # Query good-labelled VL images not yet used in a teaching session
    # (captured after the last teaching session for this material)
    db = _get_db()
    try:
        last_teach = db.execute(
            """SELECT completed_at FROM teaching_sessions
               WHERE module='tube' AND scope_key=? AND status IN ('active','superseded')
               ORDER BY completed_at DESC LIMIT 1""",
            (scope_key,),
        ).fetchone()

        since = last_teach["completed_at"] if last_teach else "1970-01-01T00:00:00+00:00"

        rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'tube'
               WHERE ci.material_id = ?
                 AND ci.module = 'tube'
                 AND ia.label = 'good'
                 AND ci.vl_path IS NOT NULL
                 AND ci.captured_at > ?
               ORDER BY ci.captured_at ASC""",
            (material_id, since),
        ).fetchall()
    finally:
        db.close()

    if not rows:
        raise HTTPException(
            status_code=409,
            detail=f"No new good-labelled tube images found for material_id='{material_id}' since last teaching."
        )

    frames = []
    for row in rows:
        frame = _cv2.imread(str(data_root / row["vl_path"]))
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise HTTPException(status_code=409, detail="No readable image files found on disk")

    teacher = get_teacher()
    teaching_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = teacher.extend(frames=frames, material_id=material_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("TubeTeacher.extend() failed for material_id=%s", material_id)
        raise HTTPException(status_code=500, detail=f"Extend failed: {e}")

    completed_at = datetime.now(timezone.utc).isoformat()

    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, completed_at,
                n_samples, model_path, threshold, extend_count, notes)
               VALUES (?, 'tube', ?, 'active', ?, ?, ?, ?, ?, ?, ?)""",
            (
                teaching_id, scope_key, started_at, completed_at,
                result["n_references_after"],
                result["template_path"],
                result["new_threshold"],
                result["extend_count"],
                f"extended with {result['n_new_samples']} new samples, "
                f"threshold {result['old_threshold']:.4f}→{result['new_threshold']:.4f}",
            ),
        )
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='tube' AND scope_key=? AND status='active'
               AND teaching_id != ?""",
            (scope_key, teaching_id),
        )
        db.commit()
    finally:
        db.close()

    # Teaching requires inspection IDLE — new templates loaded on next inspection start

    return {
        "teaching_id": teaching_id,
        "material_id": material_id,
        "n_new_samples": result["n_new_samples"],
        "n_references_before": result["n_references_before"],
        "n_references_after": result["n_references_after"],
        "old_threshold": result["old_threshold"],
        "new_threshold": result["new_threshold"],
        "extend_count": result["extend_count"],
        "extends_remaining": result["extends_remaining"],
        "completed_at": completed_at,
    }



# ============================================================================
# Teaching — Stain (PatchCore, global)
# ============================================================================

def _patch_anomalib_pandas3() -> None:
    """Monkey-patch anomalib for pandas 3.0 compatibility.

    pandas 3.0 breaks make_folder_dataset in two ways:
    1. df.loc[mask, "new_col"] = scalar — rejected when column doesn't exist
    2. df.label == DirType.NORMAL — str Enum comparison returns all False

    Must be called before any anomalib data/engine import.
    """
    import anomalib.data.datasets.image.folder as _folder_mod
    from pandas import DataFrame

    def _patched(
        normal_dir, root=None, abnormal_dir=None, normal_test_dir=None,
        mask_dir=None, split=None, extensions=None,
    ):
        from anomalib.data.utils.label import LabelName
        from anomalib.data.utils.split import Split
        from anomalib.data.utils.path import (
            DirType, _prepare_files_labels, validate_and_resolve_path,
        )
        from collections.abc import Sequence

        def _resolve(path):
            if isinstance(path, Sequence) and not isinstance(path, str):
                return [validate_and_resolve_path(p, root) for p in path]
            return [validate_and_resolve_path(path, root)] if path is not None else []

        nd = _resolve(normal_dir)
        ad = _resolve(abnormal_dir)
        ntd = _resolve(normal_test_dir)
        md = _resolve(mask_dir)

        if not nd:
            raise ValueError("normal_dir must be provided.")

        filenames, labels = [], []
        dirs = {DirType.NORMAL: nd}
        if ad:
            dirs[DirType.ABNORMAL] = ad
        if ntd:
            dirs[DirType.NORMAL_TEST] = ntd
        if md:
            dirs[DirType.MASK] = md

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

        if md and ad:
            samples["mask_path"] = ""
            samples.loc[is_abnormal, "mask_path"] = (
                samples.loc[samples.label == MASK].image_path.to_numpy()
            )
            samples["mask_path"] = samples["mask_path"].fillna("")
            samples = samples.astype({"mask_path": "str"})
        else:
            samples["mask_path"] = ""

        samples = samples.loc[
            samples.label.isin([NORMAL, ABNORMAL, NORMAL_TEST])
        ]
        samples = samples.astype({"image_path": "str"})
        samples["split"] = str(Split.TRAIN)
        samples.loc[
            samples.label.isin([ABNORMAL, NORMAL_TEST]), "split"
        ] = str(Split.TEST)
        samples.attrs["task"] = (
            "classification" if (samples["mask_path"] == "").all() else "segmentation"
        )
        if split:
            samples = samples[samples.split == str(split)].reset_index(drop=True)
        return samples

    _folder_mod.make_folder_dataset = _patched
    logger.info("Patched anomalib make_folder_dataset for pandas 3.0")


def _extract_annular_crop_for_teaching(
    frame: "np.ndarray",
    detector: "object",
) -> "np.ndarray | None":
    """Detect cone in frame and return 256x256 annular-masked crop.

    Mirrors the prepare_dataset.py pipeline used in stain-detection scripts.
    Returns None if YOLO does not detect a cone+tube in the frame.
    """
    import cv2 as _cv2
    import numpy as _np

    detections = detector.detect(frame)
    cone_bbox = None
    tube_bbox = None
    for det in detections:
        label = str(getattr(det, "label", "") or getattr(det, "class_name", "")).lower()
        bbox = getattr(det, "bbox", None) or getattr(det, "xyxy", None)
        if bbox is None:
            continue
        if "cone" in label and cone_bbox is None:
            cone_bbox = bbox
        elif "tube" in label and tube_bbox is None:
            tube_bbox = bbox

    if cone_bbox is None or tube_bbox is None:
        return None

    cx1, cy1, cx2, cy2 = (int(v) for v in cone_bbox)
    tx1, ty1, tx2, ty2 = (int(v) for v in tube_bbox)

    cone_crop = frame[cy1:cy2, cx1:cx2]
    if cone_crop.size == 0:
        return None

    # Geometry in crop coordinates
    center = (int((tx1 + tx2) / 2 - cx1), int((ty1 + ty2) / 2 - cy1))
    inner_r = float(min(tx2 - tx1, ty2 - ty1)) / 2
    outer_r = float(min(cx2 - cx1, cy2 - cy1)) / 2

    h, w = cone_crop.shape[:2]
    Y, X = _np.ogrid[:h, :w]
    dist = _np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = ((dist >= inner_r) & (dist <= outer_r)).astype(_np.uint8) * 255

    masked = cone_crop.copy()
    masked[mask == 0] = 0
    return _cv2.resize(masked, (256, 256))


def _score_crops_raw(model_pt_path: "Path", crops: list) -> list:
    """Score crops using raw PatchCore heatmap (no anomalib normalization).

    Loads model.pt directly via torch.load and disables post_processor
    normalization — same approach as sieger-parkdale-loop3-cv/StainInspector.
    Returns list of float scores (max raw anomaly within non-black pixels).
    """
    import torch as _torch
    import cv2 as _cv2
    import numpy as _np

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    ckpt = _torch.load(str(model_pt_path), map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model.eval()
    model.to(device)
    if hasattr(model, "post_processor") and model.post_processor is not None:
        model.post_processor.enable_normalization = False

    scores = []
    with _torch.no_grad():
        for crop in crops:
            rgb = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
            inp = _torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            inp = inp.to(device)
            output = model(inp)

            raw_map = getattr(output, "anomaly_map", None)
            if raw_map is None and isinstance(output, dict):
                raw_map = output.get("anomaly_map")

            if raw_map is not None:
                if hasattr(raw_map, "cpu"):
                    raw_map = raw_map.cpu().numpy()
                if raw_map.ndim == 4:
                    raw_map = raw_map[0, 0]
                elif raw_map.ndim == 3:
                    raw_map = raw_map[0]
                heatmap = _cv2.resize(raw_map.astype(_np.float32), (crop.shape[1], crop.shape[0]))
                # Non-black pixels = yarn surface (donut is already masked at 0 outside)
                donut_mask = (_cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY) > 5).astype(_np.uint8) * 255
                valid = heatmap[donut_mask > 0]
                scores.append(float(valid.max()) if len(valid) > 0 else 0.0)
            else:
                scores.append(0.0)

    return scores


def _update_teaching_session(
    teaching_id: str,
    status: str,
    *,
    model_path: str = None,
    threshold: float = None,
    completed_at: str = None,
    n_samples: int = None,
    notes: str = None,
) -> None:
    """Update a teaching_sessions row in-place (used from background tasks)."""
    from datetime import datetime, timezone

    if completed_at is None and status in ("active", "failed"):
        completed_at = datetime.now(timezone.utc).isoformat()

    fields, params = ["status = ?"], [status]
    if completed_at is not None:
        fields.append("completed_at = ?"); params.append(completed_at)
    if model_path is not None:
        fields.append("model_path = ?"); params.append(model_path)
    if threshold is not None:
        fields.append("threshold = ?"); params.append(threshold)
    if n_samples is not None:
        fields.append("n_samples = ?"); params.append(n_samples)
    if notes is not None:
        fields.append("notes = ?"); params.append(notes)
    params.append(teaching_id)

    db = _get_db()
    try:
        db.execute(
            f"UPDATE teaching_sessions SET {', '.join(fields)} WHERE teaching_id = ?",
            params,
        )
        db.commit()
    except Exception as exc:
        logger.error("teaching_sessions update failed [%s]: %s", teaching_id, exc)
    finally:
        db.close()



def _validate_stain_teaching(
    teaching_id: str,
    model_pt_path: "Path",
    threshold: float,
    data_root: "Path",
) -> dict:
    """Score good and bad annotated images against the trained stain model.

    Returns a validation report dict:
        n_good, n_bad, n_good_pass, n_bad_fail,
        true_negative_rate, true_positive_rate,
        good_scores {mean, std, max, p99},
        bad_scores  {mean, std, min, p99},
        false_positives: [image_ids that scored above threshold],
        false_negatives: [bad image_ids that scored below threshold],
        threshold, status: 'pass'|'warn'|'fail'

    Status:
        pass — TNR ≥ 0.97 and TPR ≥ 0.80 (if bad images exist)
        warn — TNR ≥ 0.90 (some false positives but workable)
        fail — TNR < 0.90 (too many false positives — do not go live)
    """
    import numpy as _np

    db = _get_db()
    try:
        good_rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'stain'
               WHERE ci.module = 'stain' AND ia.label = 'good'
                 AND ci.vl_path IS NOT NULL""",
        ).fetchall()

        bad_rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'stain'
               WHERE ci.module = 'stain' AND ia.label = 'bad'
                 AND ci.vl_path IS NOT NULL""",
        ).fetchall()
    finally:
        db.close()

    import cv2 as _cv2

    insp_cfg = _config.get("inspection", {}) if _config else {}
    yolo_weights = insp_cfg.get("weights", {}).get("visible", "weights/visible_yolo.pt")
    yolo_conf = insp_cfg.get("yolo_conf", 0.6)

    try:
        from inspection.yolo_detector import YOLODetector
        detector = YOLODetector(weights=yolo_weights, conf_threshold=yolo_conf)
    except Exception as exc:
        return {"error": f"YOLODetector load failed: {exc}", "status": "fail"}

    def _load_and_crop(rows):
        crops, ids = [], []
        for row in rows:
            frame = _cv2.imread(str(data_root / row["vl_path"]))
            if frame is None:
                continue
            crop = _extract_annular_crop_for_teaching(frame, detector)
            if crop is not None:
                crops.append(crop)
                ids.append(row["image_id"])
        return crops, ids

    good_crops, good_ids = _load_and_crop(good_rows)
    bad_crops, bad_ids = _load_and_crop(bad_rows)

    if not good_crops:
        return {"error": "No good crops could be extracted", "status": "fail"}

    try:
        good_scores_raw = _score_crops_raw(model_pt_path, good_crops)
        bad_scores_raw = _score_crops_raw(model_pt_path, bad_crops) if bad_crops else []
    except Exception as exc:
        return {"error": f"Scoring failed: {exc}", "status": "fail"}

    ga = _np.array(good_scores_raw)
    false_positive_ids = [good_ids[i] for i, s in enumerate(good_scores_raw) if s > threshold]
    n_good_pass = sum(1 for s in good_scores_raw if s <= threshold)
    tnr = n_good_pass / len(good_scores_raw) if good_scores_raw else 0.0

    bad_result = {}
    tpr = None
    false_negative_ids = []
    if bad_scores_raw:
        ba = _np.array(bad_scores_raw)
        n_bad_fail = sum(1 for s in bad_scores_raw if s > threshold)
        tpr = n_bad_fail / len(bad_scores_raw)
        false_negative_ids = [bad_ids[i] for i, s in enumerate(bad_scores_raw) if s <= threshold]
        bad_result = {
            "n_bad": len(bad_scores_raw),
            "n_bad_detected": n_bad_fail,
            "true_positive_rate": round(tpr, 4),
            "false_negatives": false_negative_ids,
            "bad_scores": {
                "min": round(float(ba.min()), 4),
                "mean": round(float(ba.mean()), 4),
                "std": round(float(ba.std()), 4),
                "p99": round(float(_np.percentile(ba, 99)), 4),
            },
        }

    # Status
    if tnr >= 0.97 and (tpr is None or tpr >= 0.80):
        status = "pass"
    elif tnr >= 0.90:
        status = "warn"
    else:
        status = "fail"

    report = {
        "teaching_id": teaching_id,
        "threshold": threshold,
        "n_good": len(good_scores_raw),
        "n_good_pass": n_good_pass,
        "true_negative_rate": round(tnr, 4),
        "false_positives": false_positive_ids,
        "good_scores": {
            "mean": round(float(ga.mean()), 4),
            "std": round(float(ga.std()), 4),
            "max": round(float(ga.max()), 4),
            "p99": round(float(_np.percentile(ga, 99)), 4),
        },
        "status": status,
        **bad_result,
    }
    return report

def _train_stain_background(
    teaching_id: str,
    image_rows: list,
    data_root: "Path",
    started_at: str,
) -> None:
    """Background task: train PatchCore stain model from captured PNG images.

    Pipeline:
        1. Load PNG images from disk (full VL frames from captured_images)
        2. YOLO detect → extract 256x256 annular cone crops (same as inference)
        3. Shuffle, split 80/20 into train/good and test/good
        4. Write MVTec-like dataset to temp dir
        5. Patch anomalib pandas3 compat, run Engine.fit + Engine.export
        6. Move model.pt to sieger_data/models/patchcore/<teaching_id>/
        7. Compute threshold: mean + 3*std of raw heatmap max on test crops
        8. Update teaching_sessions row: pending → active
        9. Supersede previous active stain session
       10. Auto-validation and save results
    """
    import os as _os
    import random
    import shutil
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch

    _os.environ["TRUST_REMOTE_CODE"] = "1"
    _torch.set_float32_matmul_precision("medium")

    logger.info("stain_teaching[%s]: background task started (%d rows)", teaching_id, len(image_rows))

    # ── Step 1: Resolve YOLO config ──────────────────────────────────────────
    insp_cfg = _config.get("inspection", {}) if _config else {}
    yolo_weights = insp_cfg.get("weights", {}).get("visible", "weights/visible_yolo.pt")
    yolo_conf = insp_cfg.get("yolo_conf", 0.6)

    try:
        from inspection.yolo_detector import YOLODetector
    except ImportError:
        import sys as _sys
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from inspection.yolo_detector import YOLODetector

    try:
        detector = YOLODetector(weights=yolo_weights, conf_threshold=yolo_conf)
    except Exception as exc:
        _update_teaching_session(teaching_id, "failed",
                                 notes=f"YOLODetector load failed: {exc}")
        return

    # ── Step 2: Load frames and extract annular crops ────────────────────────
    crops = []
    missing = []
    for image_id, vl_path in image_rows:
        fpath = data_root / vl_path
        frame = _cv2.imread(str(fpath))
        if frame is None:
            missing.append(str(fpath))
            continue
        crop = _extract_annular_crop_for_teaching(frame, detector)
        if crop is not None:
            crops.append(crop)

    if missing:
        logger.warning("stain_teaching[%s]: %d files not found on disk", teaching_id, len(missing))

    logger.info("stain_teaching[%s]: extracted %d crops from %d images",
                teaching_id, len(crops), len(image_rows))

    if len(crops) < 10:
        _update_teaching_session(
            teaching_id, "failed",
            notes=f"Only {len(crops)} valid cone crops (need ≥10). "
                  f"{len(missing)} files missing, {len(image_rows)-len(missing)-len(crops)} YOLO failures.",
        )
        return

    # ── Step 3: Shuffle and split 80/20 ─────────────────────────────────────
    random.shuffle(crops)
    split = max(int(len(crops) * 0.8), len(crops) - 2)
    train_crops = crops[:split]
    test_crops = crops[split:] or crops[-2:]

    logger.info("stain_teaching[%s]: train=%d test=%d", teaching_id, len(train_crops), len(test_crops))

    # ── Step 4-5: Train in temp dir ──────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="sieger_stain_teach_") as tmpdir:
        tmp = _Path(tmpdir)
        train_dir = tmp / "train" / "good"
        test_dir = tmp / "test" / "good"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        for i, crop in enumerate(train_crops):
            _cv2.imwrite(str(train_dir / f"train_{i:04d}.png"), crop)
        for i, crop in enumerate(test_crops):
            _cv2.imwrite(str(test_dir / f"test_{i:04d}.png"), crop)

        try:
            _patch_anomalib_pandas3()
            from anomalib.data import Folder
            from anomalib.data.utils import TestSplitMode, ValSplitMode
            from anomalib.deploy import ExportType
            from anomalib.engine import Engine
            from anomalib.models import Patchcore
        except Exception as exc:
            _update_teaching_session(teaching_id, "failed",
                                     notes=f"anomalib import failed: {exc}")
            return

        output_root = tmp / "output"
        datamodule = Folder(
            name="cone_surface",
            root=str(tmp),
            normal_dir="train/good",
            normal_test_dir="test/good",
            train_batch_size=32,
            eval_batch_size=32,
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=ValSplitMode.SAME_AS_TEST,
        )
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        engine = Engine(default_root_dir=str(output_root))

        try:
            logger.info("stain_teaching[%s]: fitting PatchCore...", teaching_id)
            engine.fit(model=model, datamodule=datamodule)
        except Exception as exc:
            logger.exception("stain_teaching[%s]: fit failed", teaching_id)
            _update_teaching_session(teaching_id, "failed",
                                     notes=f"PatchCore fit failed: {exc}")
            return

        export_root = tmp / "export"
        try:
            engine.export(
                model=model,
                export_type=ExportType.TORCH,
                export_root=str(export_root),
            )
        except Exception as exc:
            logger.exception("stain_teaching[%s]: export failed", teaching_id)
            _update_teaching_session(teaching_id, "failed",
                                     notes=f"Model export failed: {exc}")
            return

        # Find model.pt in export tree
        model_pts = list(export_root.rglob("model.pt"))
        if not model_pts:
            _update_teaching_session(teaching_id, "failed",
                                     notes="Export completed but model.pt not found.")
            return

        exported_pt = model_pts[0]

        # ── Step 6: Move model to permanent storage ──────────────────────────
        model_dest = data_root / "models" / "patchcore" / teaching_id
        weights_dst = model_dest / "weights" / "torch"
        weights_dst.mkdir(parents=True, exist_ok=True)
        for f in exported_pt.parent.iterdir():
            shutil.copy2(str(f), str(weights_dst / f.name))
        model_path = str(model_dest)
        logger.info("stain_teaching[%s]: model saved → %s", teaching_id, model_path)

        # ── Step 7: Compute threshold from raw heatmap on test crops ─────────
        # Load model.pt directly, disable post_processor normalization.
        # Threshold = mean + 3*std of raw heatmap max scores on good test crops.
        # 3-sigma rule: ~99.7% of normal cones score below threshold if
        # distribution is approx Gaussian — robust with small test sets.
        # Fallback to 0.5 if scoring fails.
        try:
            import numpy as _np_thresh
            test_scores = _score_crops_raw(weights_dst / "model.pt", test_crops)
            if test_scores:
                arr = _np_thresh.array(test_scores)
                threshold = round(float(arr.mean() + 3.0 * arr.std()), 4)
                logger.info(
                    "stain_teaching[%s]: threshold=%.4f (mean=%.4f std=%.4f n=%d)",
                    teaching_id, threshold, arr.mean(), arr.std(), len(arr),
                )
            else:
                threshold = 0.5
        except Exception as exc:
            logger.warning("stain_teaching[%s]: threshold computation failed: %s — using 0.5", teaching_id, exc)
            threshold = 0.5

        logger.info("stain_teaching[%s]: threshold=%.4f (from %d test crops)",
                    teaching_id, threshold, len(test_crops))

    # tmpdir cleaned up; model already at model_dest

    # ── Step 8: Update teaching_sessions ─────────────────────────────────────
    completed_at = datetime.now(timezone.utc).isoformat()
    _update_teaching_session(
        teaching_id, "active",
        model_path=model_path,
        threshold=threshold,
        completed_at=completed_at,
        n_samples=len(crops),
        notes=f"PatchCore trained on {len(crops)} annular crops, threshold={threshold:.4f}",
    )

    # ── Step 9: Supersede previous active stain session ─────────────────────
    db = _get_db()
    try:
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='stain' AND scope_key='global' AND status='active'
               AND teaching_id != ?""",
            (teaching_id,),
        )
        db.commit()
    finally:
        db.close()

    # ── Step 10: Auto-validation — score all good + bad annotated images ────
    import json as _json
    try:
        model_pt = _Path(model_path) / "weights" / "torch" / "model.pt"
        validation = _validate_stain_teaching(teaching_id, model_pt, threshold, data_root)
        db = _get_db()
        try:
            db.execute(
                "UPDATE teaching_sessions SET validation_json=? WHERE teaching_id=?",
                (_json.dumps(validation), teaching_id),
            )
            db.commit()
        finally:
            db.close()
        logger.info(
            "stain_teaching[%s]: validation status=%s TNR=%.3f TPR=%s",
            teaching_id, validation.get("status"),
            validation.get("true_negative_rate", 0),
            validation.get("true_positive_rate", "n/a"),
        )
    except Exception as exc:
        logger.warning("stain_teaching[%s]: validation failed: %s", teaching_id, exc)

    # Persist model_path and threshold to config.json so inspection picks
    # them up on next start (teaching requires inspection IDLE).
    import json as _json_cfg
    config_path = _Path(__file__).resolve().parent.parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = _json_cfg.load(f)
        cfg.setdefault("inspection", {})["patchcore_model"] = model_path
        cfg["inspection"]["stain_threshold"] = threshold
        with open(config_path, "w") as f:
            _json_cfg.dump(cfg, f, indent=2)
        if _config is not None:
            _config.setdefault("inspection", {})["patchcore_model"] = model_path
            _config["inspection"]["stain_threshold"] = threshold
        logger.info("stain_teaching[%s]: patchcore_model + stain_threshold written to config.json", teaching_id)
    except Exception as exc:
        logger.warning("stain_teaching[%s]: failed to persist to config.json: %s", teaching_id, exc)

    logger.info("stain_teaching[%s]: COMPLETE model=%s threshold=%.4f",
                teaching_id, model_path, threshold)


@app.post("/teaching/stain", tags=["Teaching"])
async def teaching_stain(background_tasks: BackgroundTasks):
    """Train PatchCore stain model from good-labelled captured images.

    Stain detection is GLOBAL (not per material_id) — all good-labelled VL
    images with module='stain' are pooled. One model covers all materials.

    Training is compute-heavy (PatchCore + WideResNet50). Returns immediately
    with teaching_id and status='pending'. Training runs as a background task.
    Poll GET /teaching/sessions to check status.

    Requires inspection to be IDLE (PLC c2c_start=0).
    Requires at least 10 good-labelled stain images.

    The trained model disables anomalib's internal normalization (which
    saturates scores with small test sets). Raw heatmap max is used for
    scoring, consistent with StainDetector inference.

    Teaching requires inspection to be IDLE. New model is picked up
    automatically on next inspection start.
    """
    import uuid
    from datetime import datetime, timezone

    # Check inspection is IDLE
    try:
        status = _emit_to_inspection("get_capture_status", {})
        if status.get("running"):
            raise HTTPException(
                status_code=409,
                detail="Inspection is running. Stop inspection before teaching stain model.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # Inspection service unreachable — allow teaching

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    # Query good-labelled stain VL images (global — all material IDs)
    db = _get_db()
    try:
        rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'stain'
               WHERE ci.module = 'stain'
                 AND ia.label = 'good'
                 AND ci.vl_path IS NOT NULL
               ORDER BY ci.captured_at ASC""",
        ).fetchall()
    finally:
        db.close()

    if len(rows) < 10:
        raise HTTPException(
            status_code=409,
            detail=f"Need at least 10 good-labelled stain images, found {len(rows)}. "
                   "Capture more cones and annotate before teaching.",
        )

    teaching_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Write pending row immediately so HMI can show training in progress
    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, n_samples, notes)
               VALUES (?, 'stain', 'global', 'pending', ?, ?, ?)""",
            (teaching_id, started_at, len(rows),
             f"PatchCore training queued — {len(rows)} good images"),
        )
        db.commit()
    finally:
        db.close()

    image_rows = [(row["image_id"], row["vl_path"]) for row in rows]
    background_tasks.add_task(
        _train_stain_background,
        teaching_id=teaching_id,
        image_rows=image_rows,
        data_root=data_root,
        started_at=started_at,
    )

    return {
        "teaching_id": teaching_id,
        "status": "pending",
        "n_images": len(rows),
        "message": f"PatchCore training started in background with {len(rows)} images. "
                   "Poll GET /teaching/sessions for completion.",
        "started_at": started_at,
    }



@app.get("/teaching/sessions/{teaching_id}/validate", tags=["Teaching"])
async def get_teaching_validation(teaching_id: str):
    """Return validation report for a teaching session.

    For stain: scores all good + bad annotated images against the trained model.
    Re-runs scoring live so the operator can check after adding more bad images.

    Returns:
        Validation report with true_negative_rate, true_positive_rate,
        score distributions, false_positive/negative image_ids, and
        overall status: 'pass' | 'warn' | 'fail'.
    """
    import json as _json

    db = _get_db()
    try:
        row = db.execute(
            "SELECT * FROM teaching_sessions WHERE teaching_id = ?",
            (teaching_id,),
        ).fetchone()
    finally:
        db.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Teaching session {teaching_id} not found")

    row = dict(row)
    module = row["module"]
    status = row["status"]

    if status not in ("active", "superseded"):
        # Not yet trained — return stored info
        return {
            "teaching_id": teaching_id,
            "module": module,
            "status": status,
            "message": f"Session is '{status}' — validation only available after training completes.",
            "validation": None,
        }

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    if module == "stain":
        model_path = row.get("model_path", "")
        threshold = row.get("threshold", 0.5)
        if not model_path:
            raise HTTPException(status_code=409, detail="No model path recorded for this session")
        model_pt = Path(model_path) / "weights" / "torch" / "model.pt"
        if not model_pt.exists():
            raise HTTPException(status_code=409, detail=f"Model file not found: {model_pt}")

        validation = _validate_stain_teaching(teaching_id, model_pt, threshold, data_root)

        # Persist latest validation result
        import json as _json2
        db = _get_db()
        try:
            db.execute(
                "UPDATE teaching_sessions SET validation_json=? WHERE teaching_id=?",
                (_json2.dumps(validation), teaching_id),
            )
            db.commit()
        finally:
            db.close()

        return {
            "teaching_id": teaching_id,
            "module": module,
            "model_path": model_path,
            "threshold": threshold,
            "validation": validation,
        }

    else:
        # For non-stain modules return stored validation_json if available
        stored = row.get("validation_json")
        return {
            "teaching_id": teaching_id,
            "module": module,
            "threshold": row.get("threshold"),
            "validation": _json.loads(stored) if stored else None,
            "message": f"Live re-validation not yet implemented for module '{module}'.",
        }



@app.post("/teaching/dimension", tags=["Teaching"])
async def teaching_dimension(request: Request):
    """Calibrate pixels-per-mm from captured VL images of known-size cones.

    Operator provides the known physical dimensions of the cone being imaged.
    The endpoint runs YOLO detection on all good-labelled dimension images,
    measures cone and tube bboxes (min side = inscribed diameter), and
    computes the median px/mm ratio across all valid detections.

    Outlier filtering (IQR method) removes measurements affected by extreme
    trigger jitter (cone partially out of frame or badly cropped bbox).

    Body: {
        cone_diameter_mm: float,   -- known cone outer diameter (becomes reference)
        tube_diameter_mm: float,   -- known tube outer diameter (becomes reference)
        cone_tolerance_mm: float,  -- allowed deviation for cone (default 2.0mm)
        tube_tolerance_mm: float   -- allowed deviation for tube (default 1.5mm)
    }

    Returns computed pixels_per_mm, per-source stats, and updates config.json.
    Inspection service is notified live via socket.io — no restart needed.
    """
    import uuid
    import json as _json
    from datetime import datetime, timezone

    body = await request.json()
    cone_diameter_mm = float(body.get("cone_diameter_mm", 0))
    tube_diameter_mm = float(body.get("tube_diameter_mm", 0))
    cone_tolerance_mm = float(body.get("cone_tolerance_mm", 2.0))
    tube_tolerance_mm = float(body.get("tube_tolerance_mm", 1.5))

    if cone_diameter_mm <= 0 or tube_diameter_mm <= 0:
        raise HTTPException(
            status_code=400,
            detail="cone_diameter_mm and tube_diameter_mm must be positive floats",
        )
    if cone_tolerance_mm <= 0 or tube_tolerance_mm <= 0:
        raise HTTPException(
            status_code=400,
            detail="cone_tolerance_mm and tube_tolerance_mm must be positive floats",
        )

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    # Query good-labelled dimension VL images
    db = _get_db()
    try:
        rows = db.execute(
            """SELECT ci.image_id, ci.vl_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'dimension'
               WHERE ci.module = 'dimension'
                 AND ia.label = 'good'
                 AND ci.vl_path IS NOT NULL
               ORDER BY ci.captured_at ASC""",
        ).fetchall()
    finally:
        db.close()

    if len(rows) < 5:
        raise HTTPException(
            status_code=409,
            detail=f"Need at least 5 good-labelled dimension images, found {len(rows)}.",
        )

    # Load YOLO detector
    insp_cfg = _config.get("inspection", {}) if _config else {}
    yolo_weights = insp_cfg.get("weights", {}).get("visible", "weights/visible_yolo.pt")
    yolo_conf = insp_cfg.get("yolo_conf", 0.6)

    try:
        from inspection.yolo_detector import YOLODetector
        detector = YOLODetector(weights=yolo_weights, conf_threshold=yolo_conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLODetector load failed: {exc}")

    import cv2 as _cv2
    import numpy as _np

    cone_ratios = []   # px/mm from cone bbox
    tube_ratios = []   # px/mm from tube bbox
    n_no_detection = 0

    for row in rows:
        frame = _cv2.imread(str(data_root / row["vl_path"]))
        if frame is None:
            continue

        detections = detector.detect(frame)
        cone_bbox = None
        tube_bbox = None
        for det in detections:
            label = str(getattr(det, "label", "") or getattr(det, "class_name", "")).lower()
            bbox = getattr(det, "bbox", None) or getattr(det, "xyxy", None)
            if bbox is None:
                continue
            if "cone" in label and cone_bbox is None:
                cone_bbox = bbox
            elif "tube" in label and tube_bbox is None:
                tube_bbox = bbox

        if cone_bbox is None and tube_bbox is None:
            n_no_detection += 1
            continue

        if cone_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in cone_bbox)
            cone_min_px = min(x2 - x1, y2 - y1)
            if cone_min_px > 10:  # sanity — ignore degenerate bbox
                cone_ratios.append(cone_min_px / cone_diameter_mm)

        if tube_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in tube_bbox)
            tube_min_px = min(x2 - x1, y2 - y1)
            if tube_min_px > 10:
                tube_ratios.append(tube_min_px / tube_diameter_mm)

    if not cone_ratios and not tube_ratios:
        raise HTTPException(
            status_code=409,
            detail=f"YOLO found no cone or tube detections in {len(rows)} images "
                   f"({n_no_detection} with no detection at all). "
                   "Check YOLO weights and captured images.",
        )

    def _iqr_filter(vals: list) -> list:
        """Remove outliers outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR."""
        if len(vals) < 4:
            return vals
        arr = _np.array(vals)
        q1, q3 = _np.percentile(arr, 25), _np.percentile(arr, 75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [v for v in vals if lo <= v <= hi]

    cone_filtered = _iqr_filter(cone_ratios)
    tube_filtered = _iqr_filter(tube_ratios)
    all_filtered = cone_filtered + tube_filtered

    if not all_filtered:
        raise HTTPException(
            status_code=409,
            detail="All measurements were filtered as outliers — trigger jitter too extreme. "
                   "Capture more images.",
        )

    pixels_per_mm = round(float(_np.median(all_filtered)), 4)

    # Per-source stats for the validation report
    def _stats(vals):
        if not vals:
            return None
        a = _np.array(vals)
        return {
            "n": len(vals),
            "median": round(float(_np.median(a)), 4),
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
        }

    validation = {
        "pixels_per_mm": pixels_per_mm,
        "cone_diameter_mm": cone_diameter_mm,
        "tube_diameter_mm": tube_diameter_mm,
        "cone_tolerance_mm": cone_tolerance_mm,
        "tube_tolerance_mm": tube_tolerance_mm,
        "n_images": len(rows),
        "n_no_detection": n_no_detection,
        "cone_ratios": _stats(cone_filtered),
        "cone_outliers_removed": len(cone_ratios) - len(cone_filtered),
        "tube_ratios": _stats(tube_filtered),
        "tube_outliers_removed": len(tube_ratios) - len(tube_filtered),
        "combined_n": len(all_filtered),
    }

    # Persist pixels_per_mm to config.json
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = _json.load(f)
        cfg.setdefault("inspection", {})["pixels_per_mm"] = pixels_per_mm
        cfg["inspection"]["dimension"] = {
            "cone_diameter_mm": cone_diameter_mm,
            "tube_diameter_mm": tube_diameter_mm,
            "cone_tolerance_mm": cone_tolerance_mm,
            "tube_tolerance_mm": tube_tolerance_mm,
        }
        with open(config_path, "w") as f:
            _json.dump(cfg, f, indent=2)
        # Update in-memory config
        if _config is not None:
            _config.setdefault("inspection", {})["pixels_per_mm"] = pixels_per_mm
            _config["inspection"]["dimension"] = {
                "cone_diameter_mm": cone_diameter_mm,
                "tube_diameter_mm": tube_diameter_mm,
                "cone_tolerance_mm": cone_tolerance_mm,
                "tube_tolerance_mm": tube_tolerance_mm,
            }
        logger.info(
            "Dimension teaching complete: px/mm=%.4f cone=%.1f±%.1fmm tube=%.1f±%.1fmm",
            pixels_per_mm, cone_diameter_mm, cone_tolerance_mm,
            tube_diameter_mm, tube_tolerance_mm,
        )
    except Exception as exc:
        logger.warning("Failed to persist pixels_per_mm to config.json: %s", exc)

    # Write teaching_sessions row
    teaching_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, completed_at,
                n_samples, threshold, notes, validation_json)
               VALUES (?, 'dimension', 'global', 'active', ?, ?, ?, ?, ?)""",
            (
                teaching_id, started_at, started_at,
                len(all_filtered),
                pixels_per_mm,
                f"px/mm={pixels_per_mm:.4f} from {len(all_filtered)} measurements "
                f"(cone={len(cone_filtered)} tube={len(tube_filtered)})",
                _json.dumps(validation),
            ),
        )
        # Supersede previous dimension sessions
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='dimension' AND scope_key='global' AND status='active'
               AND teaching_id != ?""",
            (teaching_id,),
        )
        db.commit()
    finally:
        db.close()

    # pixels_per_mm is written to config.json above.
    # Inspection service reads it fresh on next startup — no live reload needed
    # because teaching requires inspection to be IDLE.

    return {
        "teaching_id": teaching_id,
        "pixels_per_mm": pixels_per_mm,
        "cone_diameter_mm": cone_diameter_mm,
        "tube_diameter_mm": tube_diameter_mm,
        "cone_tolerance_mm": cone_tolerance_mm,
        "tube_tolerance_mm": tube_tolerance_mm,
        "validation": validation,
        "status": "active",
        "message": (
            f"Dimension calibration complete. "
            f"px/mm={pixels_per_mm:.4f}, "
            f"cone={cone_diameter_mm:.1f}±{cone_tolerance_mm:.1f}mm, "
            f"tube={tube_diameter_mm:.1f}±{tube_tolerance_mm:.1f}mm. "
            "Restart service to apply."
        ),
    }



@app.post("/teaching/uv", tags=["Teaching"])
async def teaching_uv(request: Request):
    """Compute UV radial_dip_threshold from good-labelled captured UV images.

    Runs the same radial log(G/B) dip pipeline used at inspection time on all
    good-labelled UV captured images. Threshold = mean + 3*std of the max_dip
    scores across all valid images.

    3-sigma rule: ~99.7% of good cones score below threshold — consistent
    with the UV detection algorithm's validated separation gap.

    Writes result to config.json under inspection.uv_inspection.radial_dip_threshold.
    Teaching requires inspection IDLE — new threshold picked up on next start.

    Returns threshold, score distribution stats, and teaching_id.
    """
    import uuid
    import json as _json
    from datetime import datetime, timezone

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    db = _get_db()
    try:
        rows = db.execute(
            """SELECT ci.image_id, ci.uv_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'uv'
               WHERE ci.module = 'uv'
                 AND ia.label = 'good'
                 AND ci.uv_path IS NOT NULL
               ORDER BY ci.captured_at ASC""",
        ).fetchall()
    finally:
        db.close()

    if len(rows) < 5:
        raise HTTPException(
            status_code=409,
            detail=f"Need at least 5 good-labelled UV images, found {len(rows)}.",
        )

    import cv2 as _cv2
    import numpy as _np

    # Instantiate UVInspection — reuses YOLO + radial dip pipeline
    uv_cfg = _config.get("inspection", {}).get("uv_inspection", {}) if _config else {}
    try:
        from inspection.uv_inspection import UVInspection
        uv_inspector = UVInspection(uv_cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"UVInspection init failed: {exc}")

    dip_scores = []
    n_failed = 0

    for row in rows:
        frame = _cv2.imread(str(data_root / row["uv_path"]))
        if frame is None:
            n_failed += 1
            continue
        result = uv_inspector.process_frame(frame)
        if result.detection_failed:
            n_failed += 1
            continue
        dip_scores.append(result.radial_dip)

    if len(dip_scores) < 5:
        raise HTTPException(
            status_code=409,
            detail=f"Only {len(dip_scores)} valid dip scores computed from {len(rows)} images "
                   f"({n_failed} YOLO/compute failures). Check UV YOLO weights.",
        )

    arr = _np.array(dip_scores)
    threshold = round(float(arr.mean() + 3.0 * arr.std()), 4)

    validation = {
        "threshold": threshold,
        "n_images": len(rows),
        "n_valid": len(dip_scores),
        "n_failed": n_failed,
        "scores": {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "p95": round(float(_np.percentile(arr, 95)), 4),
            "p99": round(float(_np.percentile(arr, 99)), 4),
        },
    }

    # Persist to config.json
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = _json.load(f)
        cfg.setdefault("inspection", {}).setdefault("uv_inspection", {})["radial_dip_threshold"] = threshold
        with open(config_path, "w") as f:
            _json.dump(cfg, f, indent=2)
        if _config is not None:
            _config.setdefault("inspection", {}).setdefault("uv_inspection", {})["radial_dip_threshold"] = threshold
        logger.info("UV radial_dip_threshold=%.4f written to config.json", threshold)
    except Exception as exc:
        logger.warning("Failed to persist UV threshold to config.json: %s", exc)

    # Write teaching_sessions row
    teaching_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, completed_at,
                n_samples, threshold, notes, validation_json)
               VALUES (?, 'uv', 'global', 'active', ?, ?, ?, ?, ?)""",
            (teaching_id, now, now, len(dip_scores), threshold,
             f"radial_dip_threshold={threshold:.4f} from {len(dip_scores)} images",
             _json.dumps(validation)),
        )
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='uv' AND scope_key='global' AND status='active'
               AND teaching_id != ?""",
            (teaching_id,),
        )
        db.commit()
    finally:
        db.close()

    return {
        "teaching_id": teaching_id,
        "threshold": threshold,
        "validation": validation,
        "status": "active",
    }


@app.post("/teaching/tail", tags=["Teaching"])
async def teaching_tail(request: Request):
    """Compute tail YOLO confidence threshold from good-labelled tail images.

    Runs YOLO tail detection on all good-labelled tail captured images.
    Threshold = mean - 2*std of detection confidence scores.

    Logic: good cones have tails present with consistent confidence. Setting
    threshold below the typical good-cone confidence means any cone where
    the tail is present will pass, while missing-tail cones (low/zero
    confidence) will fail.

    Writes result to config.json under inspection.tail_inspection.yolo_conf.
    Teaching requires inspection IDLE — new threshold picked up on next start.

    Returns threshold, score distribution stats, and teaching_id.
    """
    import uuid
    import json as _json
    from datetime import datetime, timezone

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    db = _get_db()
    try:
        rows = db.execute(
            """SELECT ci.image_id, ci.tail_path
               FROM captured_images ci
               JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = 'tail'
               WHERE ci.module = 'tail'
                 AND ia.label = 'good'
                 AND ci.tail_path IS NOT NULL
               ORDER BY ci.captured_at ASC""",
        ).fetchall()
    finally:
        db.close()

    if len(rows) < 5:
        raise HTTPException(
            status_code=409,
            detail=f"Need at least 5 good-labelled tail images, found {len(rows)}.",
        )

    import cv2 as _cv2
    import numpy as _np

    tail_cfg = _config.get("inspection", {}).get("tail_inspection", {}) if _config else {}
    try:
        from inspection.tail_inspection import TailInspection
        tail_inspector = TailInspection(tail_cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TailInspection init failed: {exc}")

    confidences = []
    n_no_tail = 0
    n_failed = 0

    for row in rows:
        frame = _cv2.imread(str(data_root / row["tail_path"]))
        if frame is None:
            n_failed += 1
            continue
        result = tail_inspector.process_frame(frame)
        if not result.model_loaded:
            n_failed += 1
            continue
        if result.tail_detected:
            confidences.append(result.confidence)
        else:
            # Good image where tail was not detected — YOLO missed it
            # Count separately, don't include in threshold computation
            n_no_tail += 1

    if len(confidences) < 5:
        raise HTTPException(
            status_code=409,
            detail=f"Only {len(confidences)} tail detections from {len(rows)} good images "
                   f"({n_no_tail} missed, {n_failed} load failures). "
                   "Check tail YOLO weights or capture more images.",
        )

    arr = _np.array(confidences)
    # Threshold = mean - 2*std: safely below typical good-cone confidence
    # Clamp to [0.1, 0.9] — avoid degenerate extremes
    threshold = round(float(max(0.1, min(0.9, arr.mean() - 2.0 * arr.std()))), 4)

    validation = {
        "threshold": threshold,
        "n_images": len(rows),
        "n_detected": len(confidences),
        "n_missed_by_yolo": n_no_tail,
        "n_load_failed": n_failed,
        "confidence_scores": {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "p5": round(float(_np.percentile(arr, 5)), 4),
        },
    }

    # Persist to config.json
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = _json.load(f)
        cfg.setdefault("inspection", {}).setdefault("tail_inspection", {})["yolo_conf"] = threshold
        with open(config_path, "w") as f:
            _json.dump(cfg, f, indent=2)
        if _config is not None:
            _config.setdefault("inspection", {}).setdefault("tail_inspection", {})["yolo_conf"] = threshold
        logger.info("Tail yolo_conf=%.4f written to config.json", threshold)
    except Exception as exc:
        logger.warning("Failed to persist tail threshold to config.json: %s", exc)

    # Write teaching_sessions row
    teaching_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db = _get_db()
    try:
        db.execute(
            """INSERT INTO teaching_sessions
               (teaching_id, module, scope_key, status, created_at, completed_at,
                n_samples, threshold, notes, validation_json)
               VALUES (?, 'tail', 'global', 'active', ?, ?, ?, ?, ?)""",
            (teaching_id, now, now, len(confidences), threshold,
             f"yolo_conf={threshold:.4f} from {len(confidences)} tail detections",
             _json.dumps(validation)),
        )
        db.execute(
            """UPDATE teaching_sessions SET status='superseded'
               WHERE module='tail' AND scope_key='global' AND status='active'
               AND teaching_id != ?""",
            (teaching_id,),
        )
        db.commit()
    finally:
        db.close()

    return {
        "teaching_id": teaching_id,
        "threshold": threshold,
        "validation": validation,
        "status": "active",
    }



# ============================================================================
# Config API — /config
# ============================================================================
# GET /config         — read full config (public)
# PUT /config/tasks   — toggle inspection modules on/off (takes effect next cone)
# PUT /config/teach   — toggle teach mode per module (takes effect next cone)
# PUT /config/shift   — shift hours (takes effect immediately)
# PUT /config/cameras — change camera settings (needs restart)
# PUT /config/plc     — change PLC settings (needs restart)
# Config is written to src/config.json on disk.
# In-memory _config is updated for sections that take effect immediately.



@app.put("/config/shift", tags=["Config"])
async def update_shift(body: dict):
    """Update shift duration.

    Takes effect immediately — no restart needed.
    The analytics system reads shift_hours from in-memory config on every cone.

    Request body example:
        {"shift_hours": 12.0}
    """
    if "shift_hours" not in body:
        raise HTTPException(status_code=400, detail="Missing 'shift_hours' in body")

    try:
        value = float(body["shift_hours"])
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="shift_hours must be a number")

    if value <= 0 or value > 24:
        raise HTTPException(status_code=400, detail="shift_hours must be between 0 and 24")

    _config["shift_hours"] = value
    _save_config()

    logger.info("Shift hours updated to %.1f", value)
    return {"ok": True, "shift_hours": value, "restart_required": False}


# ============================================================================
# Analytics — /analytics
# ============================================================================

@app.get("/analytics", tags=["Analytics"])
async def get_analytics(
    from_ts: str = None,
    to_ts: str = None,
    material_id: str = None,
):
    """Return aggregated inspection analytics.

    Without query params: returns the live in-memory shift snapshot from the
    inspection service (same data streamed via socket.io on every cone).

    With from_ts / to_ts (ISO-8601): queries SQLite for historical aggregate.
    Useful for report page on page load or date-range reports.

    Args:
        from_ts: Start timestamp ISO-8601 (e.g. 2026-03-26T06:00:00Z)
        to_ts:   End timestamp ISO-8601 (e.g. 2026-03-26T14:00:00Z)
        material_id: Filter by specific material (optional)
    """
    # Live snapshot from inspection service (no date params)
    if from_ts is None and to_ts is None and material_id is None:
        snapshot = _emit_to_service("get_analytics", {})
        if snapshot and snapshot.get("ok"):
            return snapshot.get("data", {})
        # Fallback: query SQLite for last 8 hours if service not responding
        import_ts = None

    # Historical query from SQLite
    conn = _get_db()
    try:
        where_clauses = ["trial_mode = 0"]
        params = []

        if from_ts:
            where_clauses.append("timestamp >= ?")
            params.append(from_ts)
        if to_ts:
            where_clauses.append("timestamp <= ?")
            params.append(to_ts)
        if material_id:
            where_clauses.append("material_id = ?")
            params.append(material_id)

        # If no time range given, default to last 8h
        if from_ts is None and to_ts is None:
            where_clauses.append("timestamp >= datetime('now', '-8 hours')")

        where = " AND ".join(where_clauses)

        # Overall totals
        row = conn.execute(
            f"""SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result_code=1 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN result_code=2 THEN 1 ELSE 0 END) as defect,
                SUM(CASE WHEN result_code=3 THEN 1 ELSE 0 END) as error
            FROM inspections WHERE {where}""",
            params,
        ).fetchone()

        total = row["total"] or 0
        good = row["good"] or 0
        defect = row["defect"] or 0
        error = row["error"] or 0
        rejection_pct = round(defect * 100.0 / total, 2) if total > 0 else 0.0

        # Defect breakdown by defect_type
        defect_rows = conn.execute(
            f"""SELECT defect_type, COUNT(*) as cnt
            FROM inspections
            WHERE {where} AND result_code=2
            GROUP BY defect_type""",
            params,
        ).fetchall()

        defect_breakdown = {}
        for dr in defect_rows:
            dtype = dr["defect_type"] or "unknown"
            for d in dtype.split(","):
                d = d.strip()
                if d and d != "Good":
                    defect_breakdown[d] = defect_breakdown.get(d, 0) + dr["cnt"]

        # Per-material breakdown
        mat_rows = conn.execute(
            f"""SELECT
                material_id,
                COUNT(*) as total,
                SUM(CASE WHEN result_code=1 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN result_code=2 THEN 1 ELSE 0 END) as defect
            FROM inspections WHERE {where}
            GROUP BY material_id""",
            params,
        ).fetchall()

        per_material = {
            r["material_id"]: {
                "total": r["total"],
                "good": r["good"] or 0,
                "defect": r["defect"] or 0,
                "defect_types": {},
            }
            for r in mat_rows
        }

        # Per-material defect type breakdown
        mat_defect_rows = conn.execute(
            f"""SELECT material_id, defect_type, COUNT(*) as cnt
            FROM inspections
            WHERE {where} AND result_code=2
            GROUP BY material_id, defect_type""",
            params,
        ).fetchall()

        for r in mat_defect_rows:
            mid = r["material_id"]
            if mid not in per_material:
                continue
            dtype = r["defect_type"] or "unknown"
            for d in dtype.split(","):
                d = d.strip()
                if d and d != "Good":
                    per_material[mid]["defect_types"][d] = per_material[mid]["defect_types"].get(d, 0) + r["cnt"]

        return {
            "period": {"from": from_ts, "to": to_ts},
            "shift": {
                "total": total,
                "good": good,
                "defect": defect,
                "error": error,
                "rejection_rate_pct": rejection_pct,
            },
            "defect_breakdown": defect_breakdown,
            "per_material": per_material,
        }
    finally:
        conn.close()


@app.get("/analytics/hourly", tags=["Analytics"])
async def get_analytics_hourly(
    date: str = None,
    material_id: str = None,
):
    """Return hourly Good/Defect/Error counts for a given date.

    Used for the hourly line chart on the analytics dashboard.
    Returns up to 24 rows (one per hour that had inspections).

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today.
        material_id: Filter by material (optional).

    Response:
        {
            "date": "2026-04-01",
            "hours": [
                {"hour": 6, "good": 45, "defect": 3, "error": 0},
                {"hour": 7, "good": 52, "defect": 1, "error": 0},
                ...
            ]
        }
    """
    from datetime import datetime, timezone

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = _get_db()
    try:
        where_clauses = ["timestamp LIKE ?", "trial_mode = 0"]
        params = [f"{date}%"]

        if material_id:
            where_clauses.append("material_id = ?")
            params.append(material_id)

        where = " AND ".join(where_clauses)

        rows = conn.execute(
            f"""SELECT
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                SUM(CASE WHEN result_code=1 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN result_code=2 THEN 1 ELSE 0 END) as defect,
                SUM(CASE WHEN result_code=3 THEN 1 ELSE 0 END) as error
            FROM inspections
            WHERE {where}
            GROUP BY hour
            ORDER BY hour""",
            params,
        ).fetchall()

        hours = [
            {"hour": r["hour"], "good": r["good"] or 0, "defect": r["defect"] or 0, "error": r["error"] or 0}
            for r in rows
        ]

        return {"date": date, "hours": hours}
    finally:
        conn.close()


@app.post("/analytics/reset", tags=["Analytics"])
async def reset_analytics_shift():
    """Reset the in-memory shift counters in the inspection service.

    Use at the start of a new shift if the service did not auto-reset
    (e.g. machine was running past the configured shift_hours boundary).
    """
    result = _emit_to_service("reset_analytics", {})
    if result and result.get("ok"):
        return {"ok": True, "message": "Shift counters reset"}
    return {"ok": False, "message": "Inspection service not responding — counters not reset"}

@app.get("/config", tags=["Config"])
async def get_config():
    """Return current system configuration (read-only).

    To apply config changes: edit config.json and call POST /restart.
    Always available — no auth required.
    """
    import copy
    cfg = copy.deepcopy(_config) if _config else {}
    cfg.pop("reportservice", None)
    return {"config": cfg}


def _config_path() -> Path:
    return Path(__file__).parent.parent / "config.json"


def _save_config():
    """Write current in-memory _config to disk."""
    with open(_config_path(), "w") as f:
        json.dump(_config, f, indent=4)


@app.put("/config/tasks", tags=["Config"])
async def update_tasks(body: dict):
    """Toggle inspection modules on/off.

    Controls which modules run inference (pass/fail) during inspection.
    The inspection service reads this from config.json each cycle.

    Request body example:
        {"uv_inspection": false, "tail_inspection": false}

    Valid keys: dimension_check, stain_detection, tube_pattern, uv_inspection, tail_inspection
    """
    valid_tasks = {"dimension_check", "stain_detection", "tube_pattern", "uv_inspection", "tail_inspection"}
    unknown = [k for k in body if k not in valid_tasks]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown task keys: {unknown}. Valid: {sorted(valid_tasks)}")

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    tasks = _config.setdefault("inspection", {}).setdefault("tasks", {})
    for key, value in body.items():
        tasks[key] = bool(value)

    _save_config()
    logger.info("Inspection tasks updated: %s", tasks)
    return {"ok": True, "tasks": tasks, "restart_required": False}


@app.put("/config/teach", tags=["Config"])
async def update_teach(body: dict):
    """Toggle teach mode (data capture) per module.

    When teach is ON and the module's inspection task is also ON,
    the system saves lossless numpy crops instead of running inference.
    Used for collecting good/bad images → upload to cloud → train.

    Request body example:
        {"stain_detection": true, "uv_inspection": true}

    Valid keys: stain_detection, uv_inspection, tail_inspection, dimension_check
    Note: tube_pattern is not here — tube teaches autonomously.
    """
    valid_modules = {"stain_detection", "uv_inspection", "tail_inspection", "dimension_check"}
    unknown = [k for k in body if k not in valid_modules]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown teach keys: {unknown}. Valid: {sorted(valid_modules)}")

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    teach = _config.setdefault("inspection", {}).setdefault("teach", {})
    for key, value in body.items():
        teach[key] = bool(value)

    _save_config()
    logger.info("Teach toggles updated: %s", teach)
    return {"ok": True, "teach": teach, "restart_required": False}


@app.put("/config/cameras", tags=["Config"])
async def update_cameras(body: dict):
    """Update camera exposure settings.

    Requires service restart to take effect.

    Request body example:
        {"VL": {"exposure": 12000}, "UV": {"exposure": 65000}}

    Editable fields per camera: exposure (microseconds)
    """
    valid_cameras = {"VL", "UV", "Tail"}
    unknown = [k for k in body if k not in valid_cameras]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown camera keys: {unknown}. Valid: {sorted(valid_cameras)}")

    valid_fields = {"exposure", "ip", "serial", "timeout", "trigger_debounce_us"}
    cameras = _config.setdefault("cameras", {})
    for cam_name, cam_updates in body.items():
        if not isinstance(cam_updates, dict):
            raise HTTPException(status_code=400, detail=f"Expected object for camera '{cam_name}'")
        if cam_name not in cameras:
            cameras[cam_name] = {}
        unknown_fields = [k for k in cam_updates if k not in valid_fields]
        if unknown_fields:
            raise HTTPException(status_code=400, detail=f"Unknown fields for '{cam_name}': {unknown_fields}. Valid: {sorted(valid_fields)}")
        for field, value in cam_updates.items():
            if field == "exposure":
                cameras[cam_name]["exposure"] = int(value)
            elif field == "ip":
                cameras[cam_name]["ip"] = str(value)
            elif field == "serial":
                cameras[cam_name]["serial"] = str(value)
            elif field == "timeout":
                cameras[cam_name]["timeout"] = int(value)
            elif field == "trigger_debounce_us":
                cameras[cam_name]["trigger_debounce_us"] = int(value)

    _save_config()
    logger.info("Camera config updated: %s", {k: {"exposure": v.get("exposure")} for k, v in cameras.items()})
    return {"ok": True, "cameras": {k: {"exposure": v.get("exposure")} for k, v in cameras.items()}, "restart_required": True}


@app.put("/config/plc", tags=["Config"])
async def update_plc(body: dict):
    """Update PLC connection and register settings.

    Requires service restart to take effect.

    Request body example (all fields optional — only send what you want to change):
        {
            "host": "192.168.1.110",
            "port": 502,
            "unit_id": 1,
            "timeout": 3.0,
            "poll_interval": 0.1,
            "registers": {
                "input": {
                    "sample_counter": 0,
                    "trigger": 1,
                    "c2c_start": 7,
                    "material_no": 8,
                    "basket_no": 11,
                    "loader_id": 12
                },
                "output": {
                    "result": 2,
                    "camera_error": 14,
                    "ips_status": 15,
                    "basket_no_echo": 16,
                    "material_no_echo": 17,
                    "loader_no_echo": 18,
                    "cycle_start": 9,
                    "defect_type": 19,
                    "ack": 20
                },
                "light": {
                    "uv": 4,
                    "vl": 5,
                    "yarntail": 6
                }
            }
        }
    """
    valid_top = {"host", "port", "unit_id", "timeout", "poll_interval", "registers"}
    unknown = [k for k in body if k not in valid_top]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown PLC keys: {unknown}. Valid: {sorted(valid_top)}")

    plc = _config.setdefault("plc", {})

    if "host" in body:
        plc["host"] = str(body["host"])
    if "port" in body:
        plc["port"] = int(body["port"])
    if "unit_id" in body:
        plc["unit_id"] = int(body["unit_id"])
    if "timeout" in body:
        plc["timeout"] = float(body["timeout"])
    if "poll_interval" in body:
        plc["poll_interval"] = float(body["poll_interval"])

    if "registers" in body:
        if not isinstance(body["registers"], dict):
            raise HTTPException(status_code=400, detail="'registers' must be an object")

        valid_groups = {"input", "output", "light"}
        reg_body = body["registers"]
        unknown_groups = [k for k in reg_body if k not in valid_groups]
        if unknown_groups:
            raise HTTPException(status_code=400, detail=f"Unknown register groups: {unknown_groups}. Valid: {sorted(valid_groups)}")

        registers = plc.setdefault("registers", {})

        # Merge each register group — only update fields that are sent
        for group in valid_groups:
            if group in reg_body:
                if not isinstance(reg_body[group], dict):
                    raise HTTPException(status_code=400, detail=f"'registers.{group}' must be an object")
                group_regs = registers.setdefault(group, {})
                for reg_name, reg_addr in reg_body[group].items():
                    group_regs[reg_name] = int(reg_addr)

    _save_config()
    logger.info("PLC config updated: host=%s port=%s registers=%s",
                plc.get("host"), plc.get("port"), list(plc.get("registers", {}).keys()))
    return {"ok": True, "plc": plc, "restart_required": True}


# ============================================================================
# Inspection Results — /results  (SQLite)
# ============================================================================

def _get_db() -> _sqlite3.Connection:
    """Open read-only connection to sieger.db."""
    db_path = Path(_config.get("data_root", "/home/msiegerips/sieger_data")) / "sieger.db"
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="Inspection database not initialised yet")
    conn = _sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    return conn


@app.get("/results", tags=["Results"])
async def list_results(
    material_id: str = None,
    result_code: int = None,
    date: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """List inspection results with optional filters.

    Args:
        material_id: Filter by material ID.
        result_code: 1=Good, 2=Defect, 3=Error.
        date: Filter by date prefix (YYYY-MM-DD).
        limit: Max rows to return (default 100, max 500).
        offset: Pagination offset.
    """
    limit = min(limit, 500)
    conn = _get_db()
    try:
        where = []
        params = []
        if material_id:
            where.append("material_id = ?")
            params.append(material_id)
        if result_code is not None:
            where.append("result_code = ?")
            params.append(result_code)
        if date:
            where.append("timestamp LIKE ?")
            params.append(f"{date}%")

        where_clause = "WHERE " + " AND ".join(where) if where else ""
        params += [limit, offset]

        cur = conn.execute(
            f"SELECT * FROM inspections {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {"results": rows, "count": len(rows), "offset": offset}
    finally:
        conn.close()




@app.get("/results/{inspection_id}/audit", tags=["Results"])
async def get_audit_image(inspection_id: int):
    """Serve the audit JPEG for a specific inspection.

    Returns the annotated VL frame saved at inspection time:
    - Cone crop with GOOD/DEFECT overlay
    - Per-check status (Stain, Pattern, ConeDia, TubeDia, Tail, Mixup)
    - Material ID, master ID, result label

    Used by HMI results page to show what the camera saw for each cone.
    Returns 404 if audit image was not saved (older records before audit storage).
    """
    from fastapi.responses import FileResponse

    db = _get_db()
    try:
        cur = db.execute(
            "SELECT audit_image FROM inspections WHERE id = ?",
            (inspection_id,),
        )
        row = cur.fetchone()
    finally:
        db.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")

    rel_path = row["audit_image"]
    if not rel_path:
        raise HTTPException(
            status_code=404,
            detail=f"No audit image saved for inspection {inspection_id}",
        )

    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))
    audit_path = data_root / rel_path

    if not audit_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audit image file not found on disk: {rel_path}",
        )

    return FileResponse(
        path=str(audit_path),
        media_type="image/jpeg",
        filename=f"audit_{inspection_id}.jpg",
    )

@app.get("/results/{inspection_id}", tags=["Results"])
async def get_result(inspection_id: int):
    """Get a single inspection record by ID."""
    conn = _get_db()
    try:
        cur = conn.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")
        return dict(row)
    finally:
        conn.close()


@app.get("/alerts", tags=["Results"])
async def get_alerts():
    """Active rejection rate alerts per material.

    Returns materials where rejection rate > 30% in last 20 non-trial cones.
    Used by HMI to show warning banner prompting operator action.
    """
    conn = _get_db()
    try:
        # Per-material rejection rate (last 20 non-trial cones)
        cur = conn.execute(
            """
            SELECT material_id,
                   COUNT(*) as n_total,
                   SUM(CASE WHEN result_code = 2 THEN 1 ELSE 0 END) as n_defect
            FROM (
                SELECT material_id, result_code
                FROM inspections
                WHERE trial_mode = 0
                ORDER BY id DESC
                LIMIT 200
            )
            GROUP BY material_id
            """,
        )
        alerts = []
        for row in cur.fetchall():
            if row["n_total"] >= 20:
                rate = row["n_defect"] / row["n_total"]
                if rate > 0.30:
                    alerts.append({
                        "material_id": row["material_id"],
                        "rejection_rate": round(rate, 3),
                        "n_defect": row["n_defect"],
                        "n_total": row["n_total"],
                    })
        return {"rejection_alerts": alerts, "alert_count": len(alerts)}
    except Exception:
        return {"rejection_alerts": [], "alert_count": 0}
    finally:
        conn.close()




# ============================================================================
# Reteach — operator-initiated tube reteach + teaching alert log
# ============================================================================

# In-memory ring buffer of teaching alerts (last 100)
# Populated by the inspection service via socket.io teaching_alert events
_teaching_alerts: list[dict] = []
_teaching_alerts_lock = __import__("threading").Lock()
_TEACHING_ALERTS_MAX = 100


def _on_teaching_alert(data: dict) -> None:
    """Called by socket.io client when inspection service emits teaching_alert."""
    with _teaching_alerts_lock:
        _teaching_alerts.append(data)
        if len(_teaching_alerts) > _TEACHING_ALERTS_MAX:
            _teaching_alerts.pop(0)


@app.get("/teaching/alerts", tags=["Teaching"])
async def get_teaching_alerts(limit: int = 50):
    """Return recent teaching alert events (ring buffer, newest last).

    HMI polls this to show teaching progress on the alert panel.
    Alerts include: auto-capture progress, training started/complete/failed.
    """
    with _teaching_alerts_lock:
        alerts = list(_teaching_alerts[-limit:])
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/teaching/tube/capture/start", tags=["Teaching"])
async def tube_reteach_start(request: Request):
    """Start operator-initiated tube reteach capture for a material_id.

    Adds the material_id back to the inspection service capture set.
    Resets capture counter so training re-runs on fresh images.
    Use when operator sees false rejections for an existing material.

    Body: {"material_id": "42"}
    """
    body = await request.json()
    material_id = str(body.get("material_id", "")).strip()
    if not material_id:
        raise HTTPException(status_code=400, detail="material_id required")

    try:
        _emit_to_inspection("set_capture_mode", {
            "session_id": "",
            "module": "tube",
            "material_ids": [material_id],
            "reset_counts": True,  # Reset counter for this material_id
        })
    except Exception as e:
        logger.warning("Could not notify inspection service of reteach start: %s", e)

    logger.info("Reteach capture started for material_id=%s", material_id)
    return {
        "material_id": material_id,
        "status": "capturing",
        "message": f"Reteach capture started for material {material_id}. "
                   "Images will be collected as cones arrive on the belt.",
    }


@app.post("/teaching/tube/capture/stop", tags=["Teaching"])
async def tube_reteach_stop(request: Request):
    """Stop operator-initiated tube reteach capture.

    Removes material_id from capture set. Training will have already been
    triggered automatically when min_capture_count was reached.

    Body: {"material_id": "42"}
    """
    body = await request.json()
    material_id = str(body.get("material_id", "")).strip()
    if not material_id:
        raise HTTPException(status_code=400, detail="material_id required")

    try:
        _emit_to_inspection("clear_capture_mode", {"material_id": material_id})
    except Exception as e:
        logger.warning("Could not notify inspection service of reteach stop: %s", e)

    return {
        "material_id": material_id,
        "status": "stopped",
        "message": f"Reteach capture stopped for material {material_id}.",
    }



# ============================================================================
# Cloud Upload — /cloud/upload
# Uploads captured teaching images to Azure Blob for cloud training pipeline.
# Modules: stain, uv, tail (tube trains on-device, dimension stays local)
# ============================================================================

_CLOUD_UPLOAD_MODULES = {"stain", "uv", "tail"}


@app.post("/cloud/upload", tags=["Cloud"])
async def cloud_upload(request: Request, background_tasks: BackgroundTasks):
    """Upload a teaching session's captured images to Azure Blob Storage.

    Packages all captured images for the module + session into Azure Blob:
        {customer_id}/{module}/{session_id}/
            metadata.json    ← site info, image count, config snapshot
            images/*.png     ← captured images

    Upload runs in background — returns immediately with a job_id.
    Poll GET /cloud/upload/{job_id} for status (or watch teaching_alert events).

    Body: {
        module: "stain" | "uv" | "tail",
        session_id: str   -- capture session UUID from GET /capture/sessions
    }

    Requires cloud.sas_token to be set in config.json.
    """
    import uuid as _uuid
    import copy as _copy
    from datetime import datetime, timezone

    body = await request.json()
    module = body.get("module", "").strip()
    session_id = body.get("session_id", "").strip()

    if module not in _CLOUD_UPLOAD_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"module must be one of {sorted(_CLOUD_UPLOAD_MODULES)}. "
                   "Tube teaches on-device. Dimension stays local.",
        )
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    # Check cloud config
    cloud_cfg = _config.get("cloud", {}) if _config else {}
    if not cloud_cfg.get("sas_token"):
        raise HTTPException(
            status_code=503,
            detail="cloud.sas_token not configured. "
                   "Generate a SAS token in Azure Portal and add it to config.json.",
        )

    # Fetch image paths for this session from SQLite
    db = _get_db()
    try:
        session_row = db.execute(
            "SELECT * FROM capture_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not session_row:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        image_rows = db.execute(
            """SELECT ci.image_id,
                      ci.vl_path, ci.uv_path, ci.tail_path,
                      ci.captured_at, ci.material_id,
                      ia.label
               FROM captured_images ci
               LEFT JOIN image_annotations ia
                 ON ci.image_id = ia.image_id AND ia.module = ?
               WHERE ci.session_id = ?
               ORDER BY ci.captured_at ASC""",
            (module, session_id),
        ).fetchall()
    finally:
        db.close()

    if not image_rows:
        raise HTTPException(
            status_code=409,
            detail=f"No images found for session {session_id}",
        )

    # Build image paths list — pick camera based on module
    _MODULE_CAM = {"stain": "vl_path", "uv": "uv_path", "tail": "tail_path"}
    cam_col = _MODULE_CAM[module]
    image_paths = [
        row[cam_col] for row in image_rows
        if row[cam_col] is not None
    ]

    if not image_paths:
        raise HTTPException(
            status_code=409,
            detail=f"No {cam_col} images found for session {session_id}",
        )

    # Build metadata
    insp_cfg = _copy.deepcopy(_config.get("inspection", {})) if _config else {}
    # Strip large/sensitive fields from config snapshot
    insp_cfg.pop("weights", None)
    insp_cfg.pop("patchcore_model", None)

    annotations = {
        row["image_id"]: row["label"]
        for row in image_rows
        if row["label"] is not None
    }

    metadata = {
        "customer_id": cloud_cfg.get("customer_id", "unknown"),
        "module": module,
        "session_id": session_id,
        "n_images": len(image_paths),
        "n_annotated": len(annotations),
        "annotations": annotations,
        "captured_at": dict(session_row).get("started_at", ""),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": insp_cfg,
    }

    job_id = str(_uuid.uuid4())
    data_root = Path(_config.get("data_root", "/home/msiegerips/sieger_data"))

    background_tasks.add_task(
        _cloud_upload_background,
        job_id=job_id,
        module=module,
        session_id=session_id,
        image_paths=image_paths,
        metadata=metadata,
        cloud_cfg=cloud_cfg,
        data_root=data_root,
    )

    logger.info(
        "Cloud upload job queued: job_id=%s module=%s session=%s n_images=%d",
        job_id, module, session_id, len(image_paths),
    )

    return {
        "job_id": job_id,
        "module": module,
        "session_id": session_id,
        "n_images": len(image_paths),
        "n_annotated": len(annotations),
        "status": "uploading",
        "message": f"Upload started for {len(image_paths)} {module} images. "
                   "Watch teaching_alert events for progress.",
    }


def _cloud_upload_background(
    job_id: str,
    module: str,
    session_id: str,
    image_paths: list,
    metadata: dict,
    cloud_cfg: dict,
    data_root: Path,
) -> None:
    """Background task: upload images to Azure Blob, emit teaching_alert events."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cloud.uploader import BlobUploader

    try:
        uploader = BlobUploader(cloud_cfg)
    except Exception as e:
        logger.error("BlobUploader init failed: %s", e)
        _emit_cloud_alert(module, session_id, "upload_failed", str(e), 0, len(image_paths))
        return

    _emit_cloud_alert(
        module, session_id, "uploading",
        f"Uploading {len(image_paths)} {module} images to cloud...",
        0, len(image_paths),
    )

    def on_progress(n: int, total: int) -> None:
        if n % 20 == 0 or n == total:
            _emit_cloud_alert(
                module, session_id, "uploading",
                f"Uploaded {n}/{total} images",
                n, total,
            )

    try:
        result = uploader.upload_session(
            module=module,
            session_id=session_id,
            image_paths=image_paths,
            metadata=metadata,
            data_root=data_root,
            progress_cb=on_progress,
        )

        _emit_cloud_alert(
            module, session_id, "upload_complete",
            f"Upload complete: {result['n_uploaded']}/{result['total']} images uploaded. "
            f"Training pipeline will process shortly.",
            result["n_uploaded"], result["total"],
        )
        logger.info(
            "Cloud upload complete: job_id=%s module=%s uploaded=%d failed=%d blob=%s",
            job_id, module, result["n_uploaded"], result["n_failed"], result["blob_prefix"],
        )

    except Exception as e:
        logger.exception("Cloud upload failed: job_id=%s module=%s: %s", job_id, module, e)
        _emit_cloud_alert(
            module, session_id, "upload_failed",
            f"Upload failed: {e}",
            0, len(image_paths),
        )


def _emit_cloud_alert(
    module: str,
    session_id: str,
    stage: str,
    message: str,
    count: int,
    total: int,
) -> None:
    """Reuse teaching_alert ring buffer for cloud upload progress."""
    from datetime import datetime, timezone
    alert = {
        "module": module,
        "material_id": "",
        "session_id": session_id,
        "stage": stage,
        "message": message,
        "count": count,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "cloud_upload",
    }
    with _teaching_alerts_lock:
        _teaching_alerts.append(alert)
        if len(_teaching_alerts) > _TEACHING_ALERTS_MAX:
            _teaching_alerts.pop(0)
    logger.info("CLOUD [%s] stage=%s: %s", module, stage, message)



@app.get("/", tags=["System"])
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "GHCL Yarn Cone Inspection API",
        "docs": "/docs",
        "health": "/health",
        "health_system": "/health/system",
        "health_plc": "/health/plc",
        "health_cameras": "/health/cameras",
    }
