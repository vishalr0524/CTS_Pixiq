# CLAUDE.md — pixIQ Yarn Cone Inspection

## Mode: production

## Platform: Jetson Orin NX 16GB (pixIQ)

### Hardware
- **Compute**: Jetson Orin NX 16GB (ARM64, JetPack 6.x)
- **Cameras**: Basler GigE — all identified by static IP
  - VL: a2A2600-20gcPRO (2600x2048) — 25 mm lens
  - UV: acA1920-40gc (1920x1200) — 16 mm lens
  - Tail: a2A1920-40gc (1920x1200) — 25 mm lens
- **Camera SDK**: pypylon (Basler pylon) — ARM64 requires pylon SDK installed first, then `pip install pypylon --no-binary pypylon`
- **Inference**: TensorRT FP16 via `scripts/export_tensorrt.py` (auto-detected by YOLODetector), PyTorch FP16 fallback
- **HMI**: Separate all-in-one touchscreen desktop (not on Jetson)

### Key differences from x86 (cone-transport-system)
- No IDS cameras or IDS Peak SDK — Basler only
- Hardware trigger on Line1 (not Line0)
- GrabStrategy_LatestImageOnly for auto-stale-frame discard
- pypylon manages device lifecycle internally (no Library.Initialize/Close)
- 16GB shared CPU+GPU memory — budget carefully

## Documentation Rule
**Whenever any code change is made, always update the relevant documentation** — CLAUDE.md and docs/. Every change must be reflected in `docs/CHANGELOG.md` and any affected chapters. Never leave documentation out of sync with code.

## Project Context
For full pipeline details, PLC register map, tube pattern logic, camera acquisition, and startup instructions — read `docs/project_context.md`.

## Change Log
See `docs/CHANGELOG.md`.
