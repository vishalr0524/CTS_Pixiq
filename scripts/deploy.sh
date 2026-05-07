#!/usr/bin/env bash
# ============================================================================
# Deploy Script — pixIQ Inspection System (Stage 2)
# ============================================================================
#
# Main deployment orchestrator. Run this manually on the Jetson Orin NX
# AFTER bootstrap.sh has completed successfully.
#
# Usage:
#     cd /opt/sieger/cone-transport-system-pixiq
#     ./scripts/deploy.sh                  # Full deployment (default)
#     ./scripts/deploy.sh --validate-only  # Health checks only (no changes)
#     ./scripts/deploy.sh --update         # Pull latest code & re-deploy
#     ./scripts/deploy.sh --rollback       # Restore previous state
#
# What this script does:
#     1. Pre-flight: verifies architecture, JetPack, GPU, and network
#     2. Tools: runs system_setup.sh (uv, pylon SDK, TensorRT, power mode)
#     3. DVC: configures Azure connection string and storage remote
#     4. Models: pulls .pt weight files from Azure via DVC
#     5. Python: downloads Jetson torch wheels, creates venv, runs uv sync
#     6. Config: validates / backs up src/config.json
#     7. TensorRT: exports .pt → .engine FP16 engines (~30-45 min)
#     8. Services: installs and starts sieger-inspection + sieger-api
#     9. Validation: end-to-end health checks on all services
#
# This script is IDEMPOTENT — safe to re-run after any failure.
#
# Prerequisites (set up by bootstrap.sh):
#     - git, curl, wget installed
#     - GitHub SSH key configured and authenticated
#     - Repository cloned to /opt/sieger/cone-transport-system-pixiq
#
# Reports:
#     Bootstrap:  /tmp/bootstrap_report.json        (written by bootstrap.sh)
#     Deployment: /opt/sieger/deploy_report_<ts>.json (written by this script)
#
# ============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[Deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[Deploy]${NC} $*"; }
err()  { echo -e "${RED}[Deploy]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}▶ $*${NC}"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_DIR="$SCRIPT_DIR/deployment"

REPORT_DIR="/opt/sieger"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/deployment_report_$TIMESTAMP.json"
BOOTSTRAP_REPORT="/tmp/bootstrap_report.json"
DEPLOY_FLAGS="/tmp/pixiq_deploy_flags"

# Ensure report directory exists and is writable
if [ ! -d "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR" 2>/dev/null || sudo mkdir -p "$REPORT_DIR"
fi

# Ensure report directory has proper permissions for current user
if [ ! -w "$REPORT_DIR" ]; then
    if [ "$EUID" -eq 0 ]; then
        # Running as root - set ownership to sudo user if available
        if [ -n "${SUDO_USER:-}" ]; then
            chown -R "$SUDO_USER:$SUDO_USER" "$REPORT_DIR"
        fi
    else
        # Running as non-root - try to fix permissions
        sudo chown -R "$(whoami):$(whoami)" "$REPORT_DIR" 2>/dev/null || sudo chmod 777 "$REPORT_DIR" 2>/dev/null || true
    fi
fi

# Ensure deployment flags file is writable
if [ -f "$DEPLOY_FLAGS" ]; then
    chmod 666 "$DEPLOY_FLAGS" 2>/dev/null || sudo chmod 666 "$DEPLOY_FLAGS" 2>/dev/null || true
fi

# Adjust PATH to include system and user bin directories
# /usr/local/bin is where uv is now installed (system-wide)
export PATH="$PATH:/usr/local/bin:/root/.local/bin:$HOME/.local/bin"

# Refresh command cache to find newly installed/moved tools
hash -r 2>/dev/null || true

# Mode flags
VALIDATE_ONLY=false
UPDATE_MODE=false
ROLLBACK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --validate-only) VALIDATE_ONLY=true ;;
        --update) UPDATE_MODE=true ;;
        --rollback) ROLLBACK_MODE=true ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --validate-only   Run health checks only (no deployment)"
            echo "  --update          Update existing installation"
            echo "  --rollback        Rollback to previous state"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            err "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ── Deployment State ─────────────────────────────────────────────────────────
START_TIME=$(date +%s)
START_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

declare -a PHASES=()
declare -A PHASE_STATUS=()
declare -A PHASE_CHECKS=()
declare -A PHASE_ERRORS=()
declare -A PHASE_WARNINGS=()
declare -A PHASE_DURATION=()

OVERALL_STATUS="success"

# ── Helper Functions ─────────────────────────────────────────────────────────

add_phase() {
    local phase_name=$1
    PHASES+=("$phase_name")
    PHASE_STATUS["$phase_name"]="not_started"
    PHASE_CHECKS["$phase_name"]=""
    PHASE_ERRORS["$phase_name"]=""
    PHASE_WARNINGS["$phase_name"]=""
}

start_phase() {
    local phase_name=$1
    PHASE_STATUS["$phase_name"]="in_progress"
    PHASE_START_TIME=$(date +%s)
}

end_phase() {
    local phase_name=$1
    local status=$2
    PHASE_STATUS["$phase_name"]="$status"
    
    local phase_end=$(date +%s)
    PHASE_DURATION["$phase_name"]=$((phase_end - PHASE_START_TIME))
    
    if [ "$status" == "failed" ]; then
        OVERALL_STATUS="failed"
    elif [ "$status" == "warning" ] && [ "$OVERALL_STATUS" != "failed" ]; then
        OVERALL_STATUS="warning"
    fi
}

add_check() {
    local phase=$1
    local name=$2
    local status=$3
    local details=${4:-""}
    
    local check_json="{\"name\":\"$name\",\"status\":\"$status\",\"details\":\"$details\"}"
    
    if [ -z "${PHASE_CHECKS[$phase]}" ]; then
        PHASE_CHECKS["$phase"]="$check_json"
    else
        PHASE_CHECKS["$phase"]="${PHASE_CHECKS[$phase]},$check_json"
    fi
}

add_error() {
    local phase=$1
    local message=$2
    
    if [ -z "${PHASE_ERRORS[$phase]}" ]; then
        PHASE_ERRORS["$phase"]="\"$message\""
    else
        PHASE_ERRORS["$phase"]="${PHASE_ERRORS[$phase]},\"$message\""
    fi
}

add_warning() {
    local phase=$1
    local message=$2
    
    if [ -z "${PHASE_WARNINGS[$phase]}" ]; then
        PHASE_WARNINGS["$phase"]="\"$message\""
    else
        PHASE_WARNINGS["$phase"]="${PHASE_WARNINGS[$phase]},\"$message\""
    fi
}

retry_command() {
    local max_attempts=3
    local delay=5
    local attempt=1
    local command="$@"
    
    while [ $attempt -le $max_attempts ]; do
        if eval "$command"; then
            return 0
        else
            if [ $attempt -lt $max_attempts ]; then
                warn "Attempt $attempt failed, retrying in ${delay}s..."
                sleep $delay
                delay=$((delay * 2))  # Exponential backoff
            fi
            attempt=$((attempt + 1))
        fi
    done
    
    return 1
}

check_command() {
    command -v "$1" &> /dev/null
}

# ── Generate Final Report ────────────────────────────────────────────────────

generate_report() {
    local end_time=$(date +%s)
    local end_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local duration=$((end_time - START_TIME))
    
    # Ensure report directory is writable, fallback to /tmp if not
    if [ ! -w "$REPORT_DIR" ]; then
        warn "Cannot write to $REPORT_DIR, using /tmp instead"
        REPORT_FILE="/tmp/deployment_report_$TIMESTAMP.json"
    fi
    
    mkdir -p "$(dirname "$REPORT_FILE")" 2>/dev/null || true
    
    # Get system info
    local hostname=$(hostname)
    local architecture=$(uname -m)
    local jetpack_version="unknown"
    if [ -f /etc/nv_tegra_release ]; then
        jetpack_version=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' || echo "unknown")
    fi
    
    local gpu_info="unknown"
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    fi
    
    # Reference bootstrap report as a path — kept separate (not inlined)
    local bootstrap_ref="null"
    if [ -f "$BOOTSTRAP_REPORT" ] && [ -r "$BOOTSTRAP_REPORT" ]; then
        bootstrap_ref="\"$BOOTSTRAP_REPORT\""
    fi

    # Start JSON report
    cat > "$REPORT_FILE" <<EOF
{
  "deployment_id": "deploy_$TIMESTAMP",
  "timestamp_start": "$START_TIMESTAMP",
  "timestamp_end": "$end_timestamp",
  "duration_seconds": $duration,
  "overall_status": "$OVERALL_STATUS",
  "system_info": {
    "hostname": "$hostname",
    "architecture": "$architecture",
    "jetpack_version": "$jetpack_version",
    "gpu": "$gpu_info"
  },
  "bootstrap_report": $bootstrap_ref,
  "phases": [
EOF

    
    # Add deployment phases
    local first_phase=true
    for phase in "${PHASES[@]}"; do
        [[ "$first_phase" == false ]] && echo "," >> "$REPORT_FILE"
        first_phase=false
        
        local checks="${PHASE_CHECKS[$phase]:-}"
        local errors="${PHASE_ERRORS[$phase]:-}"
        local warnings="${PHASE_WARNINGS[$phase]:-}"
        local duration="${PHASE_DURATION[$phase]:-0}"
        local status="${PHASE_STATUS[$phase]}"
        
        cat >> "$REPORT_FILE" <<EOF
    {
      "name": "$phase",
      "status": "$status",
      "duration_seconds": $duration,
      "checks": [${checks}],
      "errors": [${errors}],
      "warnings": [${warnings}]
    }
EOF
    done
    
    # Calculate validation summary
    local total_checks=0
    local passed=0
    local failed=0
    local warnings=0
    
    for phase in "${PHASES[@]}"; do
        local checks="${PHASE_CHECKS[$phase]}"
        if [ -n "$checks" ]; then
            local phase_total=$(echo "$checks" | grep -o '"status"' | wc -l)
            local phase_passed=$(echo "$checks" | grep -o '"status":"success"' | wc -l)
            local phase_failed=$(echo "$checks" | grep -o '"status":"failed"' | wc -l)
            
            total_checks=$((total_checks + phase_total))
            passed=$((passed + phase_passed))
            failed=$((failed + phase_failed))
        fi
        
        local phase_warnings="${PHASE_WARNINGS[$phase]}"
        if [ -n "$phase_warnings" ]; then
            local warning_count=$(echo "$phase_warnings" | grep -o '\"' | wc -l)
            warnings=$((warnings + warning_count / 2))
        fi
    done
    
    cat >> "$REPORT_FILE" <<EOF

  ],
  "system_info": {
    "reboot_required": $([ -f "$DEPLOY_FLAGS" ] && grep -q "REBOOT_REQUIRED=true" "$DEPLOY_FLAGS" && echo "true" || echo "false")
  },
  "validation_summary": {
    "total_checks": $total_checks,
    "passed": $passed,
    "failed": $failed,
    "warnings": $warnings
  }
}
EOF
    
    log "Deployment report saved: $REPORT_FILE"
}

# ── Deployment Header ────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║          pixIQ Deployment — Automated Installation             ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

log "Project root: $PROJECT_ROOT"
log "Deployment report: $REPORT_FILE"
echo ""

# ── Handle Special Modes ─────────────────────────────────────────────────────

if [ "$ROLLBACK_MODE" == true ]; then
    warn "Rollback mode requested"
    if [ -f "$DEPLOYMENT_DIR/rollback.sh" ]; then
        exec "$DEPLOYMENT_DIR/rollback.sh"
    else
        err "Rollback script not found: $DEPLOYMENT_DIR/rollback.sh"
        exit 1
    fi
fi

# ── Define Deployment Phases ─────────────────────────────────────────────────

add_phase "preflight"
add_phase "tools_installation"
add_phase "dvc_setup"
add_phase "authentication"
add_phase "model_files"
add_phase "python_environment"
add_phase "configuration"
add_phase "tensorrt_export"
add_phase "service_installation"
add_phase "health_validation"

# ============================================================================
# PHASE 1: PRE-FLIGHT VALIDATION
# ============================================================================

step "Phase 1: Pre-flight Validation"
start_phase "preflight"

cd "$PROJECT_ROOT"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    log "Architecture: $ARCH ✓"
    add_check "preflight" "architecture" "success" "$ARCH"
else
    warn "Architecture: $ARCH (expected aarch64)"
    add_check "preflight" "architecture" "warning" "$ARCH"
    add_warning "preflight" "Not running on ARM64 architecture"
fi

# Check JetPack
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_INFO=$(cat /etc/nv_tegra_release)
    log "JetPack: $JETPACK_INFO ✓"
    add_check "preflight" "jetpack" "success" "$JETPACK_INFO"
else
    warn "JetPack release file not found"
    add_check "preflight" "jetpack" "warning" "not detected"
    add_warning "preflight" "May not be running on Jetson device"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        log "GPU: $GPU_INFO ✓"
        add_check "preflight" "gpu" "success" "$GPU_INFO"
    else
        err "nvidia-smi found but no GPU detected"
        add_check "preflight" "gpu" "failed" "no GPU detected"
        add_error "preflight" "GPU not accessible"
    fi
else
    err "nvidia-smi not found — CUDA not accessible"
    add_check "preflight" "gpu" "failed" "nvidia-smi not found"
    add_error "preflight" "CUDA tools not available"
fi

# Check network connectivity
log "Testing network connectivity..."
if ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
    log "Internet connectivity: OK ✓"
    add_check "preflight" "internet" "success" "8.8.8.8 reachable"
else
    warn "No internet connectivity"
    add_check "preflight" "internet" "warning" "8.8.8.8 not reachable"
    add_warning "preflight" "Internet not reachable — some operations may fail"
fi

end_phase "preflight" "success"

# ============================================================================
# PHASE 2: TOOLS INSTALLATION
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 2: Tools Installation"
    start_phase "tools_installation"
    
    # Run system setup script
    SETUP_SCRIPT="$PROJECT_ROOT/scripts/system_setup.sh"
    
    if [ -f "$SETUP_SCRIPT" ]; then
        log "Running system setup script..."
        
        # Capture console output with unbuffered output
        SETUP_LOG=$(mktemp)
        # Use stdbuf to prevent the "blank hang" caused by buffering prompts
        if sudo stdbuf -oL -eL "$SETUP_SCRIPT" 2>&1 | tee "$SETUP_LOG"; then
            log "System setup completed ✓"
            
            # Check if system_setup flagged a reboot requirement
            if [ -f "$DEPLOY_FLAGS" ] && grep -q "REBOOT_REQUIRED=true" "$DEPLOY_FLAGS"; then
                warn "SYSTEM REBOOT REQUIRED for hardware performance changes to take effect."
                add_warning "tools_installation" "System reboot required for nvpmodel (MAXN) performance mode"
            fi
            
            # Add console output to report (limit to first 100 lines)
            SETUP_OUTPUT=$(cat "$SETUP_LOG" | head -n 100 | tr '\n' ' ' | tr '"' "'" )
            add_check "tools_installation" "system_setup" "success" "output: ${SETUP_OUTPUT}"
            rm -f "$SETUP_LOG"
        else
            err "System setup script failed"
            SETUP_OUTPUT=$(cat "$SETUP_LOG" | tr '\n' ' ' | tr '"' "'")
            add_check "tools_installation" "system_setup" "failed" "error: ${SETUP_OUTPUT}"
            add_error "tools_installation" "System setup script failed"
            rm -f "$SETUP_LOG"
            end_phase "tools_installation" "failed"
        fi
    else
        warn "System setup script not found: $SETUP_SCRIPT"
        add_check "tools_installation" "system_setup" "warning" "script not found"
        add_warning "tools_installation" "System setup script not found — manual setup may be required"
    fi
    
    # Install DVC with Azure support
    log "Installing DVC with Azure support..."
    if pip3 install "dvc[azure]" &> /dev/null; then
        # Refresh command cache so bash finds the new 'dvc' command immediately
        hash -r
        DVC_VERSION=$(dvc --version 2>/dev/null || echo "unknown")
        log "DVC installed: $DVC_VERSION ✓"
        add_check "tools_installation" "dvc_install" "success" "$DVC_VERSION"
    else
        err "Failed to install DVC"
        add_check "tools_installation" "dvc_install" "failed" "pip install failed"
        add_error "tools_installation" "DVC installation failed"
    fi
    
    # Check required tools
    for tool in dvc az uv systemctl; do
        if check_command "$tool"; then
            VERSION=$("$tool" --version 2>&1 | head -n1)
            log "$tool: $VERSION ✓"
            add_check "tools_installation" "$tool" "success" "$VERSION"
        else
            err "$tool not found"
            add_check "tools_installation" "$tool" "failed" "not found"
            add_error "tools_installation" "$tool is required but not installed"
        fi
    done
    
    end_phase "tools_installation" "success"
fi

# ============================================================================
# PHASE 3: DVC SETUP WITH CONNECTION STRING
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 3: DVC Configuration"
    start_phase "dvc_setup"
    
    # Verify DVC is available
    if ! check_command "dvc"; then
        err "DVC not found — should have been installed in Phase 2"
        add_check "dvc_setup" "dvc_available" "failed" "command not found"
        add_error "dvc_setup" "DVC is not available"
        end_phase "dvc_setup" "failed"
    else
        log "DVC available: $(dvc --version 2>/dev/null)"
        add_check "dvc_setup" "dvc_available" "success" "command found"
    fi
    
    # Verify DVC remote is configured from repo
    log "Verifying DVC remote configuration..."
    if dvc remote list | grep -q "azure"; then
        REMOTE_URL=$(dvc remote list | grep azure | awk '{print $2}')
        log "DVC remote found: $REMOTE_URL ✓"
        add_check "dvc_setup" "dvc_remote_list" "success" "$REMOTE_URL"
    else
        warn "DVC remote 'azure' not found"
        log "Adding DVC remote manually..."
        dvc remote add -d azure azure://dvc-store/sieger-ghcl-cv
        dvc remote modify azure account_name dhvanicvdvc
        add_check "dvc_setup" "dvc_remote_list" "warning" "added manually"
        add_warning "dvc_setup" "DVC remote was not pre-configured in repository"
    fi
    
    # Set Azure Storage connection string
    log "Configuring Azure Storage connection string..."

    _save_dvc_conn_string() {
        local cs="$1"
        if [ -d "$PROJECT_ROOT/.dvc" ]; then
            if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
                chown -R "$SUDO_USER:$SUDO_USER" "$PROJECT_ROOT/.dvc"
            elif [ "$EUID" -eq 0 ]; then
                chmod -R u+w "$PROJECT_ROOT/.dvc"
            fi
        fi
        if dvc remote modify --local azure connection_string "$cs" 2>/dev/null; then
            log "Connection string saved to .dvc/config.local ✓"
            add_check "dvc_setup" "dvc_remote_config" "success" "persisted to config.local"
        else
            warn "Failed to write to .dvc/config.local (permission issue) — env var only"
            add_check "dvc_setup" "dvc_remote_config" "warning" "env var only (not persisted)"
        fi
    }

    if [ -n "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
        # Show masked preview of existing value
        _MASKED_PREVIEW=$(echo "${AZURE_STORAGE_CONNECTION_STRING}" | head -c 40)
        warn "AZURE_STORAGE_CONNECTION_STRING is already set in environment:"
        warn "  ${_MASKED_PREVIEW}..."
        echo ""
        read -p "Keep the existing connection string? (Y/n): " KEEP_EXISTING </dev/tty
        if [[ ! "$KEEP_EXISTING" =~ ^[Nn]$ ]]; then
            log "Keeping existing connection string ✓"
            add_check "dvc_setup" "connection_string" "success" "from environment (kept)"
        else
            unset AZURE_STORAGE_CONNECTION_STRING
            log "Will prompt for a new connection string."
        fi
    fi

    # If not set (or user chose to replace), enter interactive re-prompt loop
    if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
        warn "AZURE_STORAGE_CONNECTION_STRING not set — entering connection string setup."
        warn "Format: DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
        echo ""
        while true; do
            read -p "Enter Azure Storage connection string: " CONN_STRING </dev/tty
            echo ""
            if [ -z "$CONN_STRING" ]; then
                warn "No connection string entered."
                read -p "Skip connection string setup? DVC pull will FAIL without it. (y/N): " SKIP_CS </dev/tty
                if [[ "$SKIP_CS" =~ ^[Yy]$ ]]; then
                    add_check "dvc_setup" "connection_string" "warning" "skipped by operator"
                    add_warning "dvc_setup" "Connection string not configured — DVC pull will fail"
                    break
                fi
                continue
            fi
            # Show masked preview for confirmation
            _PREVIEW=$(echo "$CONN_STRING" | head -c 40)
            echo "  Preview: ${_PREVIEW}..."
            read -p "Use this connection string? (y/n/re-enter): " CONFIRM </dev/tty
            echo ""
            if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
                export AZURE_STORAGE_CONNECTION_STRING="$CONN_STRING"
                log "Connection string confirmed and exported ✓"
                add_check "dvc_setup" "connection_string" "success" "manually entered and confirmed"
                # Offer persistence
                read -p "Save to .dvc/config.local for future runs? (y/N): " SAVE_CONFIG </dev/tty
                if [[ "$SAVE_CONFIG" =~ ^[Yy]$ ]]; then
                    _save_dvc_conn_string "$AZURE_STORAGE_CONNECTION_STRING"
                else
                    log "Connection string available for this session only."
                    add_check "dvc_setup" "dvc_remote_config" "success" "env var set (not persisted)"
                fi
                break
            elif [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
                log "Re-entering connection string..."
                continue
            else
                log "Re-entering connection string..."
                continue
            fi
        done
    fi
    
    end_phase "dvc_setup" "success"
fi

# ============================================================================
# PHASE 4: MODEL FILES (DVC PULL)
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 4: Pulling Model Files"
    start_phase "model_files"
    
    if check_command "dvc"; then
        log "Running dvc pull..."
        
        log "Verifying connection string is set before pulling..."
        if [ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ]; then
            err "AZURE_STORAGE_CONNECTION_STRING not set — cannot authenticate with Azure"
            add_error "model_files" "Azure connection string is required for DVC pull"
            end_phase "model_files" "failed"
        else
            log "Starting DVC pull from Azure — this may take several minutes..."
            echo ""
            echo -e "  ${CYAN}[▶] Pulling model files from Azure DVC remote...${NC}"
            echo ""

            DVC_PULL_LOG="/tmp/dvc_pull_${TIMESTAMP}.log"
            DVC_PULL_EXIT=0

            # Launch dvc pull in background; tail its log live
            ( dvc pull --verbose 2>&1 | tee "$DVC_PULL_LOG" ) &
            DVC_PULL_PID=$!

            # Heartbeat ticker: print elapsed time every 10 s so the
            # operator can see progress instead of a blank pause
            _PULL_START=$(date +%s)
            while kill -0 "$DVC_PULL_PID" 2>/dev/null; do
                sleep 10
                _elapsed=$(( $(date +%s) - _PULL_START ))
                printf "  ${YELLOW}[⏳] dvc pull still running — %ds elapsed...${NC}\n" "$_elapsed"
            done

            # Collect exit status
            wait "$DVC_PULL_PID" || DVC_PULL_EXIT=$?

            echo ""
            if [ "$DVC_PULL_EXIT" -eq 0 ]; then
                log "DVC pull completed successfully ✓"
                add_check "model_files" "dvc_pull" "success" "completed"

                # Validate model files
                for model in visible_yolo uv_yolo yarn_tail_v3; do
                    MODEL_FILE="weights/${model}.pt"
                    if [ -f "$MODEL_FILE" ]; then
                        SIZE=$(du -h "$MODEL_FILE" | awk '{print $1}')
                        log "$MODEL_FILE: $SIZE ✓"
                        add_check "model_files" "$model" "success" "size: $SIZE"
                    else
                        err "$MODEL_FILE not found after dvc pull"
                        add_check "model_files" "$model" "failed" "file missing"
                        add_error "model_files" "$MODEL_FILE is required but missing"
                    fi
                done

                end_phase "model_files" "success"
            else
                err "DVC pull failed (exit code $DVC_PULL_EXIT)"
                err "Log: $DVC_PULL_LOG"
                add_check "model_files" "dvc_pull" "failed" "exit $DVC_PULL_EXIT — see $DVC_PULL_LOG"
                add_error "model_files" "DVC pull failed — check Azure credentials and $DVC_PULL_LOG"
                end_phase "model_files" "failed"
            fi
        fi
    else
        err "DVC not available"
        add_check "model_files" "dvc" "failed" "not found"
        add_error "model_files" "DVC is required"
        end_phase "model_files" "failed"
    fi
    
    end_phase "model_files" "success"
fi

# ============================================================================
# PHASE 5: PYTHON ENVIRONMENT
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 5: Python Environment Setup"
    start_phase "python_environment"
    
    if check_command "uv"; then
        # ── Download Jetson torch wheels if not already present ──────────────────
        # pyproject.toml references these wheel files with [tool.uv.sources]
        # They are too large for git — downloaded from SharePoint on first deploy.
        TORCH_WHL="torch-2.4.0a0+07cecf4168.nv24.5-cp310-cp310-linux_aarch64.whl"
        TORCHVISION_WHL="torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
        # ⚠ These URLs are intentionally hardcoded. Update them here if the wheel changes.
        TORCH_URL="https://dhvani365-my.sharepoint.com/:u:/g/personal/vishal_dhvaniai_com/IQD_p8YR82VgTYogrlmPnnPtAaLmzskVt_M3CtUxge9WFps?download=1"
        TORCHVISION_URL="https://dhvani365-my.sharepoint.com/:u:/g/personal/vishal_dhvaniai_com/IQD9rTvbhnyURbxfCkw5HqSsAS62Sx4jEUO5OwyvrgLZkQ4?download=1"

        log "Checking Jetson PyTorch wheel files..."
        for whl_file in "$TORCH_WHL" "$TORCHVISION_WHL"; do
            if [ -f "$PROJECT_ROOT/$whl_file" ]; then
                log "  $whl_file already present ✓"
                add_check "python_environment" "${whl_file%%.*}_wheel" "success" "already present"
            else
                if [ "$whl_file" = "$TORCH_WHL" ]; then
                    _WHL_URL="$TORCH_URL"
                else
                    _WHL_URL="$TORCHVISION_URL"
                fi
                log "  Downloading $whl_file ..."
                if wget --show-progress -q -O "$PROJECT_ROOT/$whl_file" "$_WHL_URL"; then
                    SIZE=$(du -h "$PROJECT_ROOT/$whl_file" | awk '{print $1}')
                    log "  $whl_file downloaded ($SIZE) ✓"
                    add_check "python_environment" "${whl_file%%.*}_wheel" "success" "downloaded $SIZE"
                else
                    err "  Failed to download $whl_file"
                    add_check "python_environment" "${whl_file%%.*}_wheel" "failed" "download failed"
                    add_error "python_environment" "Could not download $whl_file — check internet access"
                fi
            fi
        done
        echo ""

        # ── Virtual environment ────────────────────────────────────────────────────
        # Detect system Python 3.10 explicitly (Jetson requirement)
        SYSTEM_PYTHON310=""
        for py_candidate in python3.10 /usr/bin/python3.10 /usr/local/bin/python3.10; do
            if command -v "$py_candidate" &>/dev/null; then
                SYSTEM_PYTHON310="$py_candidate"
                break
            fi
        done
        
        if [ -z "$SYSTEM_PYTHON310" ]; then
            err "Python 3.10 not found — Jetson Orin NX requires Python 3.10.x for CUDA/TensorRT"
            add_check "python_environment" "python310_check" "failed" "python 3.10 not found"
            add_error "python_environment" "Python 3.10 not found on system"
            end_phase "python_environment" "failed"
            return 1
        fi
        
        PYTHON_VERSION=$($SYSTEM_PYTHON310 --version 2>&1 | awk '{print $2}')
        log "Using system Python: $SYSTEM_PYTHON310 (version $PYTHON_VERSION)"
        add_check "python_environment" "python310_check" "success" "$PYTHON_VERSION at $SYSTEM_PYTHON310"
        
        # Create or update virtual environment
        if [ -d ".venv" ]; then
            log "Virtual environment exists — removing to ensure Python 3.10 compatibility..."
            rm -rf .venv
            add_check "python_environment" "venv_cleanup" "success" "removed existing venv"
        fi
        
        log "Creating virtual environment with Python 3.10 and system-site-packages..."
        if uv venv --python "$SYSTEM_PYTHON310" --system-site-packages; then
            log "Virtual environment created ✓"
            add_check "python_environment" "venv_created" "success" "python $PYTHON_VERSION with system-site-packages"
            
            # Verify the venv is using the correct Python version
            VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
            if [[ "$VENV_PYTHON_VERSION" =~ ^3\.10\. ]]; then
                log "Virtual environment Python version verified: $VENV_PYTHON_VERSION ✓"
                add_check "python_environment" "venv_python_version" "success" "$VENV_PYTHON_VERSION"
            else
                err "Virtual environment is using Python $VENV_PYTHON_VERSION instead of 3.10.x"
                err "This will cause CUDA/TensorRT compatibility issues"
                add_check "python_environment" "venv_python_version" "failed" "$VENV_PYTHON_VERSION (expected 3.10.x)"
                add_error "python_environment" "Virtual environment using wrong Python version"
                end_phase "python_environment" "failed"
                return 1
            fi
        else
            err "Failed to create virtual environment"
            add_check "python_environment" "venv_created" "failed" "uv venv failed"
            add_error "python_environment" "Virtual environment creation failed"
            end_phase "python_environment" "failed"
        fi
        
        log "Syncing dependencies with uv..."
        if uv sync; then
            log "Dependencies synced ✓"
            add_check "python_environment" "dependencies" "success" "uv sync completed"
        else
            err "Dependency sync failed"
            add_check "python_environment" "dependencies" "failed" "uv sync failed"
            add_error "python_environment" "Failed to sync Python dependencies"
            end_phase "python_environment" "failed"
        fi
        
        # Validate critical imports using uv run (ensures proper environment setup)
        log "Validating critical imports..."
        
        for module in pypylon tensorrt torch; do
            if uv run python -c "import $module" 2>/dev/null; then
                log "$module: OK ✓"
                add_check "python_environment" "${module}_import" "success" "importable"
            else
                err "$module: Import failed"
                add_check "python_environment" "${module}_import" "failed" "import error"
                add_error "python_environment" "Failed to import $module"
            fi
        done
    else
        err "uv not found"
        add_check "python_environment" "uv" "failed" "not found"
        add_error "python_environment" "uv package manager not available"
        end_phase "python_environment" "failed"
    fi
    
    end_phase "python_environment" "success"
fi

# ============================================================================
# PHASE 6: CONFIGURATION
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 6: Site Configuration"
    start_phase "configuration"
    
    CONFIG_FILE="$PROJECT_ROOT/src/config.json"
    CONFIG_TEMPLATE="$PROJECT_ROOT/deploy/config.template.json"
    
    if [ -f "$CONFIG_FILE" ]; then
        log "Configuration file exists: $CONFIG_FILE"
        add_check "configuration" "config_exists" "success" "$CONFIG_FILE"
        
        # Backup existing config
        BACKUP_FILE="$CONFIG_FILE.backup.$TIMESTAMP"
        cp "$CONFIG_FILE" "$BACKUP_FILE"
        log "Backup created: $BACKUP_FILE"
        add_check "configuration" "config_backup" "success" "$BACKUP_FILE"
    else
        if [ -f "$CONFIG_TEMPLATE" ]; then
            warn "Configuration file not found, will need manual setup"
            log "Template available at: $CONFIG_TEMPLATE"
            add_check "configuration" "config_exists" "warning" "not found"
            add_warning "configuration" "config.json not found — manual configuration required"
        else
            err "Configuration template not found"
            add_check "configuration" "config_template" "failed" "not found"
            add_error "configuration" "Configuration template missing"
        fi
    fi
    
    # Note: Interactive configuration would go here
    # For now, we just validate the file exists
    
    end_phase "configuration" "success"
fi

# ============================================================================
# PHASE 7: TENSORRT EXPORT
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 7: TensorRT Model Export"
    start_phase "tensorrt_export"
    
    EXPORT_SCRIPT="$PROJECT_ROOT/scripts/export_tensorrt.py"
    
    if [ -f "$EXPORT_SCRIPT" ] && check_command "uv"; then
        log "Exporting TensorRT engines (this may take several minutes)..."
        
        if uv run python "$EXPORT_SCRIPT"; then
            log "TensorRT export completed ✓"
            add_check "tensorrt_export" "export" "success" "completed"
            
            # Validate engine files
            for model in visible_yolo uv_yolo yarn_tail_v3; do
                ENGINE_FILE="weights/${model}.engine"
                if [ -f "$ENGINE_FILE" ]; then
                    SIZE=$(du -h "$ENGINE_FILE" | awk '{print $1}')
                    log "$ENGINE_FILE: $SIZE ✓"
                    add_check "tensorrt_export" "$model" "success" "size: $SIZE"
                else
                    warn "$ENGINE_FILE not found"
                    add_check "tensorrt_export" "$model" "warning" "file missing"
                    add_warning "tensorrt_export" "$ENGINE_FILE was not created"
                fi
            done
        else
            warn "TensorRT export had errors"
            add_check "tensorrt_export" "export" "warning" "completed with errors"
            add_warning "tensorrt_export" "Some models may not have been exported"
        fi
    else
        warn "Cannot run TensorRT export (script or venv missing)"
        add_check "tensorrt_export" "export" "warning" "skipped"
        add_warning "tensorrt_export" "TensorRT export skipped"
    fi
    
    end_phase "tensorrt_export" "success"
fi

# ============================================================================
# PHASE 8: SERVICE INSTALLATION
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 8: Service Installation"
    start_phase "service_installation"
    
    SERVICE_DIR="$PROJECT_ROOT/deploy/systemd"
    NGINX_DIR="$PROJECT_ROOT/deploy/nginx"
    
    # Detect deployment environment
    DEPLOY_USER=$(whoami)
    DEPLOY_GROUP=$(id -gn)
    WORKDIR="$PROJECT_ROOT"
    VENV_PYTHON="$WORKDIR/.venv/bin/python"
    
    log "Deployment environment detected:"
    log "  User: $DEPLOY_USER"
    log "  Group: $DEPLOY_GROUP"
    log "  WorkDir: $WORKDIR"
    log "  Python: $VENV_PYTHON"
    
    # Validate environment
    if ! id "$DEPLOY_USER" &>/dev/null; then
        err "User $DEPLOY_USER does not exist"
        add_check "service_installation" "user_validation" "failed" "user not found"
        add_error "service_installation" "Deployment user does not exist"
        end_phase "service_installation" "failed"
    else
        add_check "service_installation" "user_validation" "success" "$DEPLOY_USER exists"
    fi
    
    if [ ! -x "$VENV_PYTHON" ]; then
        err "Python venv not found at $VENV_PYTHON"
        add_check "service_installation" "venv_validation" "failed" "venv python missing"
        add_error "service_installation" "Virtual environment Python not found"
        end_phase "service_installation" "failed"
    else
        add_check "service_installation" "venv_validation" "success" "$VENV_PYTHON exists"
    fi
    
    if [ ! -d "$WORKDIR" ]; then
        err "Working directory not found: $WORKDIR"
        add_check "service_installation" "workdir_validation" "failed" "directory missing"
        add_error "service_installation" "Working directory does not exist"
        end_phase "service_installation" "failed"
    else
        add_check "service_installation" "workdir_validation" "success" "$WORKDIR exists"
    fi
    
    if [ -d "$SERVICE_DIR" ]; then
        log "Generating systemd service files from templates..."
        
        # Process each service template
        for service_template in "$SERVICE_DIR"/sieger-*.service; do
            SERVICE_NAME=$(basename "$service_template")
            SERVICE_FILE="/tmp/$SERVICE_NAME"
            
            log "Processing $SERVICE_NAME..."
            
            # Replace placeholders with actual values
            sed -e "s|{{DEPLOY_USER}}|$DEPLOY_USER|g" \
                -e "s|{{DEPLOY_GROUP}}|$DEPLOY_GROUP|g" \
                -e "s|{{WORKDIR}}|$WORKDIR|g" \
                -e "s|{{VENV_PYTHON}}|$VENV_PYTHON|g" \
                "$service_template" > "$SERVICE_FILE"
            
            # Copy to systemd directory
            sudo cp "$SERVICE_FILE" /etc/systemd/system/
            log "$SERVICE_NAME generated and installed ✓"
            rm -f "$SERVICE_FILE"
        done
        
        # Reload systemd
        sudo systemctl daemon-reload
        log "systemd daemon reloaded ✓"
        add_check "service_installation" "systemd_reload" "success" "daemon reloaded"
        
        # Enable and start services
        for service in sieger-inspection sieger-api; do
            # Enable service
            if sudo systemctl enable "$service"; then
                log "$service: Enabled ✓"
            else
                err "$service: Enable failed"
                add_error "service_installation" "Failed to enable $service"
            fi
            
            # Start service
            if sudo systemctl start "$service"; then
                log "$service: Started ✓"
                add_check "service_installation" "$service" "success" "enabled and started"
            else
                err "$service: Start failed"
                add_check "service_installation" "$service" "failed" "start failed"
                add_error "service_installation" "Failed to start $service"
            fi
        done
    else
        err "Service directory not found: $SERVICE_DIR"
        add_check "service_installation" "services" "failed" "directory missing"
        add_error "service_installation" "Service files not found"
        end_phase "service_installation" "failed"
    fi
    
    # Install nginx configuration
    if [ -d "$NGINX_DIR" ]; then
        log "Installing nginx configuration..."
        
        if [ -f "$NGINX_DIR/sieger.conf" ]; then
            sudo cp "$NGINX_DIR/sieger.conf" /etc/nginx/sites-enabled/
            log "nginx config copied to /etc/nginx/sites-enabled/ ✓"
            
            # Test nginx configuration
            if sudo nginx -t; then
                log "nginx configuration test passed ✓"
                add_check "service_installation" "nginx_config" "success" "valid"
                
                # Restart nginx
                if sudo systemctl restart nginx; then
                    log "nginx restarted ✓"
                    add_check "service_installation" "nginx_restart" "success" "active"
                else
                    err "nginx restart failed"
                    add_check "service_installation" "nginx_restart" "failed" "restart error"
                    add_error "service_installation" "Failed to restart nginx"
                fi
            else
                err "nginx configuration test failed"
                add_check "service_installation" "nginx_config" "failed" "invalid"
                add_error "service_installation" "nginx configuration is invalid"
            fi
        else
            warn "nginx configuration file not found: $NGINX_DIR/sieger.conf"
            add_check "service_installation" "nginx_config" "warning" "file not found"
            add_warning "service_installation" "nginx configuration not installed"
        fi
    else
        warn "nginx directory not found: $NGINX_DIR"
        add_check "service_installation" "nginx_dir" "warning" "directory missing"
        add_warning "service_installation" "nginx configuration directory missing"
    fi
    
    end_phase "service_installation" "success"
fi

# ============================================================================
# PHASE 9: HEALTH VALIDATION
# ============================================================================

step "Phase 9: Health Validation"
start_phase "health_validation"

log "Waiting for services to start (10s)..."
sleep 10

# Check API health
log "Checking API health..."
if curl -s -f http://localhost:5002/health > /dev/null; then
    log "API health: OK ✓"
    add_check "health_validation" "api_health" "success" "http://localhost:5002/health"
else
    err "API health check failed"
    add_check "health_validation" "api_health" "failed" "endpoint unreachable"
    add_error "health_validation" "API is not responding"
fi

# Check system health
log "Checking system health..."
HEALTH_RESPONSE=$(curl -s http://localhost:5002/health/system 2>/dev/null || echo "{}")

if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
    HEALTH_STATUS=$(echo "$HEALTH_RESPONSE" | grep -oP '"status":\s*"\K[^"]+' || echo "unknown")
    log "System health: $HEALTH_STATUS"
    add_check "health_validation" "system_health" "success" "$HEALTH_STATUS"
else
    warn "Could not retrieve system health"
    add_check "health_validation" "system_health" "warning" "unavailable"
    add_warning "health_validation" "System health endpoint not responding"
fi

# Check service status
for service in sieger-inspection sieger-api; do
    if systemctl is-active --quiet "$service"; then
        log "$service: Active ✓"
        add_check "health_validation" "${service}_status" "success" "active"
    else
        err "$service: Not active"
        add_check "health_validation" "${service}_status" "failed" "inactive"
        add_error "health_validation" "$service is not running"
    fi
done

end_phase "health_validation" "success"

# ============================================================================
# DEPLOYMENT COMPLETE
# ============================================================================

step "Deployment Complete"

# Generate final report
generate_report

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# ── Operator-Friendly Final Summary ─────────────────────────────────────────
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
if [ "$OVERALL_STATUS" == "success" ]; then
    echo -e "${CYAN}║${NC}  ${GREEN}✅  DEPLOYMENT COMPLETED SUCCESSFULLY${NC}                              ${CYAN}║${NC}"
elif [ "$OVERALL_STATUS" == "warning" ]; then
    echo -e "${CYAN}║${NC}  ${YELLOW}⚠️   DEPLOYMENT COMPLETED WITH WARNINGS${NC}                             ${CYAN}║${NC}"
else
    echo -e "${CYAN}║${NC}  ${RED}❌  DEPLOYMENT FAILED${NC}                                               ${CYAN}║${NC}"
fi
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Elapsed:${NC} ${TOTAL_DURATION}s"
echo -e "  ${BLUE}Report: ${NC} $REPORT_FILE"
echo ""

# ── Phase Summary Table ───────────────────────────────────────────────────────
echo -e "${CYAN}┌──────────────────────────────┬────────────┬──────────┐${NC}"
echo -e "${CYAN}│  Phase                       │  Status    │  Time    │${NC}"
echo -e "${CYAN}├──────────────────────────────┼────────────┼──────────┤${NC}"
for phase in "${PHASES[@]}"; do
    p_status="${PHASE_STATUS[$phase]}"
    p_dur="${PHASE_DURATION[$phase]:-─}s"
    # Count errors and warnings for this phase
    p_err_cnt=0
    p_warn_cnt=0
    if [ -n "${PHASE_ERRORS[$phase]:-}" ]; then
        p_err_cnt=$(echo "${PHASE_ERRORS[$phase]}" | grep -o '"' | wc -l)
        p_err_cnt=$(( p_err_cnt / 2 ))
    fi
    if [ -n "${PHASE_WARNINGS[$phase]:-}" ]; then
        p_warn_cnt=$(echo "${PHASE_WARNINGS[$phase]}" | grep -o '"' | wc -l)
        p_warn_cnt=$(( p_warn_cnt / 2 ))
    fi
    # Status icon + colour
    if [ "$p_status" == "success" ]; then
        STATUS_COL="${GREEN}✓ ok         ${NC}"
    elif [ "$p_status" == "warning" ]; then
        STATUS_COL="${YELLOW}⚠ warn ($p_warn_cnt)  ${NC}"
    elif [ "$p_status" == "failed" ]; then
        STATUS_COL="${RED}✗ FAILED ($p_err_cnt)${NC}"
    else
        STATUS_COL="${YELLOW}○ skipped    ${NC}"
    fi
    printf "${CYAN}│${NC}  %-28s ${CYAN}│${NC}  %b${CYAN}│${NC}  %-6s  ${CYAN}│${NC}\n" \
        "$phase" "$STATUS_COL" "$p_dur"
done
echo -e "${CYAN}└──────────────────────────────┴────────────┴──────────┘${NC}"
echo ""

# ── Per-Check Checklist ───────────────────────────────────────────────────────
echo -e "${CYAN}Detailed Check Results:${NC}"
for phase in "${PHASES[@]}"; do
    p_status="${PHASE_STATUS[$phase]}"
    if [ "$p_status" == "success" ]; then
        ph_icon="${GREEN}✓${NC}"
    elif [ "$p_status" == "warning" ]; then
        ph_icon="${YELLOW}⚠${NC}"
    elif [ "$p_status" == "failed" ]; then
        ph_icon="${RED}✗${NC}"
    else
        ph_icon="${YELLOW}○${NC}"
    fi
    echo -e "  $ph_icon ${BLUE}[$phase]${NC}"
    # Parse JSON-like checks string for individual check names/statuses
    checks_raw="${PHASE_CHECKS[$phase]:-}"
    if [ -n "$checks_raw" ]; then
        # Each check is {"name":"...","status":"...","details":"..."}
        # Extract name+status pairs with simple grep
        while IFS= read -r check_entry; do
            c_name=$(echo "$check_entry"  | grep -oP '"name":\s*"\K[^"]+' || echo "?")
            c_status=$(echo "$check_entry" | grep -oP '"status":\s*"\K[^"]+' || echo "?")
            c_detail=$(echo "$check_entry" | grep -oP '"details":\s*"\K[^"]+' || echo "")
            if [ "$c_status" == "success" ]; then
                c_icon="${GREEN}  ✓${NC}"
            elif [ "$c_status" == "warning" ]; then
                c_icon="${YELLOW}  ⚠${NC}"
            elif [ "$c_status" == "failed" ]; then
                c_icon="${RED}  ✗${NC}"
            else
                c_icon="${YELLOW}  ○${NC}"
            fi
            if [ -n "$c_detail" ]; then
                echo -e "    $c_icon $c_name — $c_detail"
            else
                echo -e "    $c_icon $c_name"
            fi
        done < <(echo "$checks_raw" | grep -oP '\{[^}]+\}')
    fi
    # Print errors for this phase (if any)
    if [ -n "${PHASE_ERRORS[$phase]:-}" ]; then
        echo "${PHASE_ERRORS[$phase]}" | grep -oP '"\K[^"]+(?=")' | while read -r emsg; do
            echo -e "      ${RED}ERROR:${NC} $emsg"
        done
    fi
done
echo ""

# ── Next Actions Block ────────────────────────────────────────────────────────
NEXT_ACTIONS=()
if [ -f "$DEPLOY_FLAGS" ] && grep -q "REBOOT_REQUIRED=true" "$DEPLOY_FLAGS"; then
    NEXT_ACTIONS+=("⚡ REBOOT REQUIRED — nvpmodel MAXN mode needs a reboot: sudo reboot")
fi
for phase in "${PHASES[@]}"; do
    if [ "${PHASE_STATUS[$phase]}" == "failed" ]; then
        NEXT_ACTIONS+=("🔴 Phase '$phase' FAILED — review: $REPORT_FILE")
    fi
done
if [ -f "$PROJECT_ROOT/src/config.json" ]; then
    if grep -q 'YOUR_CAMERA_IP\|PLACEHOLDER\|CHANGEME' "$PROJECT_ROOT/src/config.json" 2>/dev/null; then
        NEXT_ACTIONS+=("📝 config.json still has placeholder values — edit src/config.json")
    fi
fi

if [ ${#NEXT_ACTIONS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Next Actions Required:${NC}"
    for action in "${NEXT_ACTIONS[@]}"; do
        echo -e "  $action"
    done
    echo ""
fi

# ── Final exit ────────────────────────────────────────────────────────────────
if [ "$OVERALL_STATUS" != "success" ]; then
    err "Deployment had issues. Review the report: $REPORT_FILE"
    exit 1
fi

echo -e "${GREEN}Services running. Operator commands:${NC}"
echo "  sudo systemctl status sieger-inspection sieger-api"
echo "  journalctl -u sieger-inspection -f"
echo "  cat $REPORT_FILE"
echo ""
exit 0
