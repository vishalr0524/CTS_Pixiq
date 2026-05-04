## Phase - 1 Early deployment setting on the Jetson for deployment:
 
| Tool | Check Command | Purpose |
|---|---|---|
| `git` | `git --version` | Clone/pull the repo |
| `dvc` | `dvc --version` | Pull model files |
| `az` CLI | `az --version` | Azure Blob access |
| `python3` / `uv` | `uv --version` | Virtualenv + dependencies |
| `systemctl` | `systemctl --version` | Service management |
 
Install DVC with Azure support if not already present:
```bash
pip install "dvc[azure]"
```
 
---
 
## Step 1 — Clone the Repository
 
```bash
git clone -b bugfix git@github.com:dhvani-cv/cone-transport-system-pixiq.git
cd cone-transport-system-pixiq
```
 
> ✅ Confirmed working. Clone the `bugfix` branch onto the device.
 
---
 
## Step 2 — Configure DVC and Pull Model Files from Azure Blob
 
### 2a. Verify tools
```bash
dvc --version
az --version
```
 
### 2b. Authenticate with Azure
```bash
az login
```
 
### 2c. Get the Azure Storage Account Key
```bash
az storage account keys list \
  --account-name dhvanicvdvc \
  --query "[0].value" -o tsv
```
Copy the key from the output.
 
### 2d. Set the DVC remote key (local config — not committed to git)
```bash
dvc remote modify --local azure account_key <paste-key-here>
```
 
### 2e. Pull model files
```bash
dvc pull
```
 
> ✅ After pull, verify that model files referenced in `.dvc` files exist on disk and are non-empty.
 
---
 
## Step 3 — Run Setup Script and Export TensorRT Files
 
```bash
# Run system setup (installs dependencies, configures device)
sudo ./scripts/system_setup.sh
 
# Create virtual environment with Python 3.10 (with system site packages for Jetson CUDA/TensorRT access)
uv venv --python 3.10 --system-site-packages
 
# Install project dependencies
uv sync
 
# Export TensorRT engine files from model files
uv run python scripts/export_tensorrt.py
```
 
> ⚠️ `export_tensorrt.py` can take several minutes on the Jetson. Do not interrupt it.
 
---
 
## Step 4 — Install and Enable Systemd Services
 
### 4a. Copy service files to systemd
```bash
sudo cp deploy/systemd/sieger-inspection.service /etc/systemd/system/
sudo cp deploy/systemd/sieger-api.service /etc/systemd/system/
```
 
### 4b. Reload daemon and enable services (start on boot)
```bash
sudo systemctl daemon-reload
sudo systemctl enable sieger-inspection sieger-api
```
 
### 4c. Start services
```bash
sudo systemctl start sieger-inspection sieger-api
```
 
### 4d. Verify services are running
```bash
sudo systemctl status sieger-inspection sieger-api
```
 
Both services should show `active (running)`. If either shows `failed`, check logs (Step 5).
 
---
 
## Step 5 — Check Logs
 
Follow live logs for each service:
 
```bash
# Inspection service logs
journalctl -u sieger-inspection -f
 
# API service logshere is a critical change since there should be this "Unreleased — 2026-04-14
Cameras: Updated Tail camera model from acA1920-40gc (ace classic) to a2A1920-40gc (ace 2). Added lens specs: VL 25mm, UV 16mm, Tail 25mm. Updated all docs, diagrams, README, CLAUDE.md, and source docstrings.
Diagrams: Added inspection_sequence.mmd (capture sequence diagram) and system_block_diagram.mmd (full system block diagram) to docs/diagrams/. Corrected camera models and added lens specs in all .mmd files.
Docs overhaul: Purged all stale x86/IDS references from documentation. Updated project_context.md, 01_system_overview.md, 02_architecture.md, 04_camera_capture.md, 05_inspection_pipeline.md, 10_rest_api.md, 13_configuration.md, 14_deployment.md, 17_pixiq_setup.md, index.md, api_reference.md, frontend_v3_guide.md. All docs now reflect pixIQ as primary platform: Jetson Orin NX 16GB, Basler GigE cameras (pypylon), TensorRT FP16, 192.168.1.0/24 subnet, Line1 hardware trigger. Removed all IDS Peak, RTX 3050, 192.168.2.x, and "x86 as primary" references.
Mermaid diagrams: Replaced all ASCII art diagrams with inline Mermaid across 01_system_overview.md, 02_architecture.md, 04_camera_capture.md, 05_inspection_pipeline.md, 06_tube_pattern.md, 07_stain_detection.md, 11_socketio_service.md, 14_deployment.md, project_context.md. Embedded system block diagram and inspection sequence from .mmd files inline in system overview and project context. Updated 00_index.md diagram index." so take out from ## Unreleased — 2026-04-29 and remaining changes be updated under this Unreleased — 2026-04-29, so that there will be Unreleased — 2026-04-29, Unreleased — 2026-04-14 and Unreleased — 2026-04-08 ok

journalctl -u sieger-api -f
```
 
---
 
## Useful Commands (Quick Reference)
 
```bash
# Stop both services
sudo systemctl stop sieger-inspection sieger-api
 
# Restart both services
sudo systemctl restart sieger-inspection sieger-api
 
# Check status at any time
sudo systemctl status sieger-inspection sieger-api
```
 
---

## Phase 2 - Automated Deployment — Two-Stage Script (IMPLEMENTED)

### Problem: Bootstrap Challenge
Fresh Jetson devices don't have `git` or other tools installed. Non-technical users need a simple way to get started.

### Solution: Two-Stage Deployment (✅ Ready for Use)

**Status**: Scripts created and ready for deployment testing.

**Implementation Files**:
- 📄 [`scripts/bootstrap.sh`](scripts/bootstrap.sh) — Bootstrap script (Stage 1)
- 📄 [`scripts/deploy.sh`](scripts/deploy.sh) — Main deployment orchestrator (Stage 2)
- 📄 [`scripts/deployment/rollback.sh`](scripts/deployment/rollback.sh) — Rollback on failure

#### **Stage 1: Bootstrap Script** (`bootstrap.sh`)
**Location**: Outside the git repository (can't clone without git!)

**Hosting Options**:
1. **GitHub Gist** (Recommended): Public gist accessible via `wget`
   - URL: `https://gist.github.com/dhvani-cv/{hash}/raw/bootstrap.sh`
   - Easy to update, version controlled
2. **Azure Blob Storage**: Same account (`dhvanicvdvc`) with public read access
3. **USB Drive**: For offline/air-gapped deployments at `/media/usb/bootstrap.sh`

**Bootstrap Responsibilities**:
- Install minimal prerequisites: `git`, `curl`, `wget`, `python3-pip`
- Guide GitHub SSH key setup with interactive prompts
- Validate authentication: `ssh -T git@github.com`
- Clone repository to `/opt/sieger/cone-transport-system-pixiq`
- Launch main deployment script: `scripts/deploy.sh`
- Generate bootstrap report: `/tmp/bootstrap_report.json`

**Idempotency (Safe to Re-run)**:
- ✅ Checks if git/curl/wget already installed before attempting installation
- ✅ Skips SSH key generation if `~/.ssh/id_ed25519` already exists
- ✅ Uses `git clone` on fresh install, `git pull` if repo already exists
- ✅ Updates existing installation without duplicating work
- ⚠️ Re-running after failed bootstrap is safe and recommended

**User Instructions** (Copy-Paste Simple):
```bash
# Online deployment
wget https://gist.github.com/dhvani-cv/{hash}/raw/bootstrap.sh && bash bootstrap.sh

# Offline deployment (USB)
bash /media/usb/bootstrap.sh
```

---

#### **Stage 2: Main Deployment Script** (`scripts/deploy.sh`)
**Location**: Inside git repository at `scripts/deploy.sh`

**Idempotency (Safe to Re-run)**:
- ✅ **System packages**: apt-get checks if already installed before installing
- ✅ **Python venv**: Checks if `.venv` exists before creating, uses `uv sync` to update dependencies
- ✅ **DVC remote config**: `dvc remote modify` is idempotent (overwrites existing)
- ✅ **Model files**: `dvc pull` skips already-downloaded files (checks checksums)
- ✅ **TensorRT export**: Checks if `.engine` files exist and valid before re-exporting
- ✅ **Systemd services**: `cp`, `enable`, `start` are all idempotent
- ✅ **Config generation**: Backs up existing `config.json` before overwriting
- ⚠️ **Network config**: GigE tuning (sysctl) and MTU changes are reapplied (safe)
- 🔄 **Updates supported**: Re-running after `git pull` updates code and redeploys safely

**Deployment Phases** (Sequential with JSON validation):

1. **Pre-flight Validation**
   - Check architecture (`aarch64`)
   - Verify JetPack installed (`/etc/nv_tegra_release`)
   - Check GPU accessible (`nvidia-smi`)
   - Validate network connectivity (internet, factory LAN)
   - Tool version checks (dvc, az, uv, systemctl)

2. **System Tools Installation**
   - Run `scripts/system_setup.sh` (GigE tuning, pylon SDK, system packages)
   - Install DVC with Azure support
   - Install Azure CLI if missing

3. **Authentication Setup**
   - Validate GitHub SSH (already done by bootstrap)
   - Check Azure CLI authenticated (`az account show`)
   - Configure DVC remote with Azure storage key
   - Prompt for manual `az login` if needed (browser-based, can't automate)

4. **Model Files (DVC Pull)**
   - Pull model files: `dvc pull`
   - Validate checksums against `.dvc` files
   - Verify all `.pt` files exist and are non-zero
   - Retry logic for transient network failures

5. **Python Environment Setup**
   - Create virtual environment: `uv venv --python 3.10 --system-site-packages`
   - Install dependencies: `uv sync`
   - Validate critical imports: `pypylon`, `tensorrt`, `torch`

6. **Site Configuration**
   - Interactive config generator: `scripts/deployment/configure.py`
   - Prompt for: PLC IP, 3 camera IPs, Azure SAS token, site_id, data_root
   - Validate inputs: ping tests, IP format checks
   - Generate `src/config.json` from `deploy/config.template.json`
   - Support `--env-file` for automated/CI deployments

7. **TensorRT Model Export**
   - Run `scripts/export_tensorrt.py`
   - Export all 3 YOLO models to `.engine` files
   - Run warm-up inference to validate engines
   - Report per-model status (exported, inference_time_ms, file_size_mb)

8. **Service Installation**
   - Copy systemd service files to `/etc/systemd/system/`
   - Reload systemd daemon
   - Enable services for auto-start
   - Start services
   - Wait for "active" state (with timeout + retries)

9. **Health Validation**
   - Poll API health endpoints: `/health`, `/health/system`, `/health/plc`, `/health/cameras`
   - Validate PLC connection
   - Validate all 3 cameras connected
   - Validate models loaded in inspection service
   - Retry with exponential backoff

10. **Final Reporting**
    - Merge bootstrap report (if exists)
    - Generate comprehensive deployment report
    - Save to: `/opt/sieger/deployment_report_YYYY-MM-DD_HHMMSS.json`
    - Display human-readable summary
    - Save deployment logs to `logs/deployment.log`

---

### Deployment Report File Locations

| Report File | Location | Purpose | Retention |
|-------------|----------|---------|-----------|
| **Bootstrap Report** | `/tmp/bootstrap_report.json` | Pre-repo validation, tool installation | Temporary - merged into main report |
| **Main Deployment Report** | `/opt/sieger/deployment_report_YYYY-MM-DD_HHMMSS.json` | Complete deployment validation | Permanent reference |
| **Deployment Logs** | `logs/deployment.log` | Human-readable deployment logs | Permanent (rotated) |
| **Deployment JSON Logs** | `logs/deployment.json.log` | Structured logs (Azure Monitor format) | Permanent (rotated) |

**Report Schema** (JSON Structure):
```json
{
  "deployment_id": "deploy_20260504_143022",
  "timestamp_start": "2026-05-04T14:30:22Z",
  "timestamp_end": "2026-05-04T14:45:18Z",
  "duration_seconds": 896,
  "overall_status": "success|warning|failed",
  "system_info": {
    "hostname": "jetson-orin-ghcl-01",
    "architecture": "aarch64",
    "jetpack_version": "6.0",
    "gpu": "Orin NX 16GB",
    "cuda_version": "12.2",
    "tensorrt_version": "8.6.1"
  },
  "phases": [
    {
      "name": "bootstrap",
      "status": "success",
      "duration_seconds": 45,
      "timestamp_start": "2026-05-04T14:30:22Z",
      "timestamp_end": "2026-05-04T14:31:07Z",
      "checks": [
        {"name": "git_installed", "status": "success", "version": "2.34.1"},
        {"name": "ssh_key_configured", "status": "success", "details": "github.com SSH OK"},
        {"name": "repo_cloned", "status": "success", "branch": "bugfix", "commit": "abc123"}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "tools_installation",
      "status": "success",
      "duration_seconds": 180,
      "checks": [
        {"name": "dvc_installed", "status": "success", "version": "3.48.4"},
        {"name": "azure_cli_installed", "status": "success", "version": "2.60.0"},
        {"name": "uv_installed", "status": "success", "version": "0.1.24"},
        {"name": "pylon_sdk", "status": "success", "version": "7.4.0"}
      ],
      "errors": [],
      "warnings": ["pylon SDK already installed, skipped reinstall"]
    },
    {
      "name": "authentication",
      "status": "success",
      "duration_seconds": 30,
      "checks": [
        {"name": "github_ssh", "status": "success"},
        {"name": "azure_cli_auth", "status": "success", "account": "user@dhvani.com"},
        {"name": "dvc_remote", "status": "success"}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "model_files",
      "status": "success",
      "duration_seconds": 120,
      "checks": [
        {"name": "dvc_pull", "status": "success", "files_pulled": 6},
        {"name": "visible_yolo.pt", "status": "success", "size_mb": 5.5, "checksum_valid": true},
        {"name": "uv_yolo.pt", "status": "success", "size_mb": 5.5, "checksum_valid": true},
        {"name": "yarn_tail_v3.pt", "status": "success", "size_mb": 5.5, "checksum_valid": true}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "python_environment",
      "status": "success",
      "duration_seconds": 90,
      "checks": [
        {"name": "venv_created", "status": "success", "python_version": "3.10.12"},
        {"name": "dependencies_installed", "status": "success", "packages": 85},
        {"name": "pypylon_import", "status": "success"},
        {"name": "tensorrt_import", "status": "success"},
        {"name": "torch_import", "status": "success", "cuda_available": true}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "configuration",
      "status": "success",
      "duration_seconds": 60,
      "checks": [
        {"name": "config_generated", "status": "success", "path": "src/config.json"},
        {"name": "plc_ping", "status": "success", "ip": "192.168.1.110"},
        {"name": "camera_vl_ping", "status": "success", "ip": "192.168.1.160"},
        {"name": "camera_uv_ping", "status": "success", "ip": "192.168.1.161"},
        {"name": "camera_tail_ping", "status": "success", "ip": "192.168.1.162"}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "tensorrt_export",
      "status": "success",
      "duration_seconds": 240,
      "checks": [
        {
          "name": "visible_yolo.engine",
          "status": "success",
          "exported": true,
          "inference_time_ms": 15.2,
          "file_size_mb": 11.2,
          "warmup_passed": true
        },
        {
          "name": "uv_yolo.engine",
          "status": "success",
          "exported": true,
          "inference_time_ms": 14.8,
          "file_size_mb": 11.2,
          "warmup_passed": true
        },
        {
          "name": "yarn_tail_v3.engine",
          "status": "success",
          "exported": true,
          "inference_time_ms": 16.1,
          "file_size_mb": 11.2,
          "warmup_passed": true
        }
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "service_installation",
      "status": "success",
      "duration_seconds": 30,
      "checks": [
        {"name": "sieger-inspection.service", "status": "active", "enabled": true},
        {"name": "sieger-api.service", "status": "active", "enabled": true}
      ],
      "errors": [],
      "warnings": []
    },
    {
      "name": "health_validation",
      "status": "success",
      "duration_seconds": 45,
      "checks": [
        {"name": "api_health", "status": "success", "endpoint": "http://localhost:5002/health"},
        {"name": "system_health", "status": "healthy", "endpoint": "/health/system"},
        {"name": "plc_health", "status": "connected", "host": "192.168.1.110"},
        {
          "name": "cameras_health",
          "status": "ok",
          "all_connected": true,
          "cameras": [
            {"name": "vl", "connected": true, "ip": "192.168.1.160"},
            {"name": "uv", "connected": true, "ip": "192.168.1.161"},
            {"name": "tail", "connected": true, "ip": "192.168.1.162"}
          ]
        },
        {"name": "models_loaded", "status": "success", "yolo": true, "patchcore": true}
      ],
      "errors": [],
      "warnings": []
    }
  ],
  "validation_summary": {
    "total_checks": 45,
    "passed": 45,
    "failed": 0,
    "warnings": 2
  }
}
```

---

### Code Structure

```
scripts/
├── deploy.sh                    # Main deployment orchestrator (bash)
├── deployment/
│   ├── __init__.py
│   ├── validators.py            # Validation functions (system, tools, auth, health)
│   ├── configure.py             # Interactive config.json generator
│   ├── reporter.py              # JSON report generation and aggregation
│   └── rollback.sh              # Rollback to previous state on failure
└── export_tensorrt.py           # Enhanced with JSON output

deploy/
├── config.template.json         # Template for src/config.json
├── deployment_report_schema.json  # JSON schema for validation reports
└── nginx/
    └── sieger.conf

docs/
└── automated_deployment.md      # User-facing deployment guide
```

---

### Error Handling & Rollback

**Automatic Rollback Triggers**:
- Critical tool installation failure (git, dvc, az, uv)
- Authentication failure (GitHub, Azure)
- All model files failed to download
- TensorRT export failed for all models
- Both services failed to start
- Health validation failed (no PLC, no cameras, no models)

**Rollback Actions** (`scripts/deployment/rollback.sh`):
- Stop services
- Restore previous git commit (if updating)
- Restore config.json backup
- Clean up partial installation
- Generate rollback report: `/opt/sieger/rollback_report_YYYY-MM-DD_HHMMSS.json`

**Manual Rollback**:
```bash
./scripts/deploy.sh --rollback
```

---

### Retry Logic

**Transient Failures** (automatic retry with exponential backoff):
- Azure CLI connection timeouts
- DVC pull network errors
- API health check (service starting up)
- PLC connection (network fluctuation)

**Max Retries**: 3 attempts with 5s, 10s, 20s delays

---

### Deployment Validation Cross-Check

Non-technical users can cross-check deployment success by reviewing the JSON report:

**Quick Validation Checklist**:
1. `overall_status` = `"success"` ✅
2. All phase `status` = `"success"` ✅
3. `validation_summary.failed` = `0` ✅
4. Services: `sieger-inspection.service` and `sieger-api.service` = `"active"` ✅
5. Health: `system_health.status` = `"healthy"` ✅
6. Cameras: `cameras_health.all_connected` = `true` ✅
7. PLC: `plc_health.status` = `"connected"` ✅

**If any check fails**, the report includes:
- Detailed error messages
- Suggested troubleshooting steps
- Rollback instructions

---

### Next Steps

1. ✅ **Bootstrap script created** — [`scripts/bootstrap.sh`](scripts/bootstrap.sh)
2. ✅ **Main deployment script created** — [`scripts/deploy.sh`](scripts/deploy.sh) with 9 phases
3. ✅ **Rollback script created** — [`scripts/deployment/rollback.sh`](scripts/deployment/rollback.sh)
4. 🔲 **Host bootstrap.sh** on GitHub Gist or Azure Blob for external access
5. 🔲 **Create Python modules** in `scripts/deployment/`:
   - `validators.py` — System, tool, auth, health validation functions
   - `configure.py` — Interactive config.json generator with prompts
   - `reporter.py` — JSON report aggregation and generation
6. 🔲 **Enhance `scripts/export_tensorrt.py`** to output JSON status per model
7. 🔲 **Create `deploy/config.template.json`** from existing config.json
8. 🔲 **Write `docs/automated_deployment.md`** user guide with:
   - Copy-paste commands for deployment
   - Screenshots of terminal output
   - Troubleshooting common issues
   - JSON report interpretation guide
9. 🔲 **Test on fresh Jetson Orin NX** with non-technical user simulation
10. 🔲 **Update CHANGELOG.md** with automated deployment feature

---

### How to Use (Once Hosted)

**For Non-Technical Users**:
```bash
# Single command deployment (online)
wget https://gist.github.com/dhvani-cv/{hash}/raw/bootstrap.sh && bash bootstrap.sh

# Or from USB drive (offline)
bash /media/usb/bootstrap.sh
```

**The bootstrap script automatically**:
1. Installs git and required tools
2. Guides SSH key setup for GitHub
3. Clones the repository
4. Launches main deployment (`scripts/deploy.sh`)
5. Deploys services with health validation
6. Generates deployment report

**No manual intervention required** except:
- Adding SSH key to GitHub (one-time, guided)
- Entering site-specific values (PLC IP, camera IPs, Azure SAS token)

---