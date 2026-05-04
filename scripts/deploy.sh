#!/usr/bin/env bash
# ============================================================================
# Main Deployment Script — pixIQ (Stage 2)
# ============================================================================
#
# Comprehensive deployment orchestrator for pixIQ yarn cone inspection system.
# This script handles the full deployment after bootstrap completes.
#
# Usage:
#     ./scripts/deploy.sh                 # Full deployment
#     ./scripts/deploy.sh --update        # Update existing installation
#     ./scripts/deploy.sh --rollback      # Rollback to previous state
#     ./scripts/deploy.sh --validate-only # Only run health checks
#
# This script is IDEMPOTENT — safe to re-run after failures.
#
# Prerequisites (installed by bootstrap.sh):
#     - git, curl, wget
#     - GitHub SSH access configured
#     - Repository cloned to /opt/sieger/cone-transport-system-pixiq
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
    
    mkdir -p "$REPORT_DIR"
    
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
  "phases": [
EOF
    
    # Add bootstrap phase if report exists
    if [ -f "$BOOTSTRAP_REPORT" ]; then
        cat "$BOOTSTRAP_REPORT" >> "$REPORT_FILE"
        echo "," >> "$REPORT_FILE"
    fi
    
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
        if sudo "$SETUP_SCRIPT"; then
            log "System setup completed ✓"
            add_check "tools_installation" "system_setup" "success" "completed"
        else
            err "System setup script failed"
            add_check "tools_installation" "system_setup" "failed" "script error"
            add_error "tools_installation" "System setup script failed"
            end_phase "tools_installation" "failed"
        fi
    else
        warn "System setup script not found: $SETUP_SCRIPT"
        add_check "tools_installation" "system_setup" "warning" "script not found"
        add_warning "tools_installation" "System setup script not found — manual setup may be required"
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
# PHASE 3: AUTHENTICATION
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 3: Authentication Setup"
    start_phase "authentication"
    
    # Check GitHub SSH (already validated by bootstrap)
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        log "GitHub SSH: OK ✓"
        add_check "authentication" "github_ssh" "success" "authenticated"
    else
        err "GitHub SSH authentication failed"
        add_check "authentication" "github_ssh" "failed" "not authenticated"
        add_error "authentication" "GitHub SSH access required"
    fi
    
    # Check Azure CLI authentication
    if check_command "az"; then
        if az account show &> /dev/null; then
            AZURE_ACCOUNT=$(az account show --query user.name -o tsv 2>/dev/null || echo "unknown")
            log "Azure CLI: Authenticated as $AZURE_ACCOUNT ✓"
            add_check "authentication" "azure_cli" "success" "$AZURE_ACCOUNT"
        else
            warn "Azure CLI not authenticated"
            warn "Run 'az login' to authenticate"
            add_check "authentication" "azure_cli" "warning" "not authenticated"
            add_warning "authentication" "Azure CLI not authenticated — DVC pull may fail"
        fi
    fi
    
    # Configure DVC remote
    if check_command "dvc"; then
        log "Configuring DVC remote..."
        
        # Check if DVC remote already configured
        if dvc remote list | grep -q "azure"; then
            log "DVC remote already configured ✓"
            add_check "authentication" "dvc_remote" "success" "already configured"
        else
            log "DVC remote 'azure' not found in dvc remote list"
            add_check "authentication" "dvc_remote" "warning" "not configured"
            add_warning "authentication" "DVC remote not configured"
        fi
        
        # Try to get Azure storage key if authenticated
        if az account show &> /dev/null; then
            log "Retrieving Azure storage account key..."
            if STORAGE_KEY=$(az storage account keys list --account-name dhvanicvdvc --query "[0].value" -o tsv 2>/dev/null); then
                dvc remote modify --local azure account_key "$STORAGE_KEY"
                log "DVC remote key configured ✓"
                add_check "authentication" "dvc_key" "success" "configured"
            else
                warn "Failed to retrieve Azure storage key"
                add_check "authentication" "dvc_key" "warning" "retrieval failed"
                add_warning "authentication" "Could not retrieve Azure storage key automatically"
            fi
        fi
    fi
    
    end_phase "authentication" "success"
fi

# ============================================================================
# PHASE 4: MODEL FILES (DVC PULL)
# ============================================================================

if [ "$VALIDATE_ONLY" == false ]; then
    step "Phase 4: Pulling Model Files"
    start_phase "model_files"
    
    if check_command "dvc"; then
        log "Running dvc pull..."
        
        if retry_command "dvc pull"; then
            log "DVC pull completed ✓"
            add_check "model_files" "dvc_pull" "success" "completed"
            
            # Validate model files
            for model in visible_yolo uv_yolo yarn_tail_v3; do
                MODEL_FILE="weights/${model}.pt"
                if [ -f "$MODEL_FILE" ]; then
                    SIZE=$(du -h "$MODEL_FILE" | awk '{print $1}')
                    log "$MODEL_FILE: $SIZE ✓"
                    add_check "model_files" "$model" "success" "size: $SIZE"
                else
                    err "$MODEL_FILE not found"
                    add_check "model_files" "$model" "failed" "file missing"
                    add_error "model_files" "$MODEL_FILE is required"
                fi
            done
        else
            err "DVC pull failed"
            add_check "model_files" "dvc_pull" "failed" "pull failed"
            add_error "model_files" "DVC pull failed — check Azure authentication"
            end_phase "model_files" "failed"
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
        # Create or update virtual environment
        if [ -d ".venv" ]; then
            log "Virtual environment exists, syncing dependencies..."
            add_check "python_environment" "venv_exists" "success" "found"
        else
            log "Creating virtual environment..."
            uv venv --python 3.10 --system-site-packages
            add_check "python_environment" "venv_created" "success" "python 3.10"
        fi
        
        log "Installing dependencies..."
        if uv sync; then
            log "Dependencies installed ✓"
            add_check "python_environment" "dependencies" "success" "uv sync completed"
        else
            err "Dependency installation failed"
            add_check "python_environment" "dependencies" "failed" "uv sync failed"
            add_error "python_environment" "Failed to install Python dependencies"
            end_phase "python_environment" "failed"
        fi
        
        # Validate critical imports
        log "Validating critical imports..."
        
        PYTHON_EXEC=".venv/bin/python"
        
        for module in pypylon tensorrt torch; do
            if $PYTHON_EXEC -c "import $module" 2>/dev/null; then
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
    
    if [ -f "$EXPORT_SCRIPT" ] && [ -d ".venv" ]; then
        log "Exporting TensorRT engines (this may take several minutes)..."
        
        if .venv/bin/python "$EXPORT_SCRIPT"; then
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
    
    if [ -d "$SERVICE_DIR" ]; then
        log "Installing systemd services..."
        
        # Copy service files
        sudo cp "$SERVICE_DIR"/sieger-*.service /etc/systemd/system/
        sudo systemctl daemon-reload
        
        for service in sieger-inspection sieger-api; do
            # Enable service
            if sudo systemctl enable "$service"; then
                log "$service: Enabled ✓"
            else
                err "$service: Enable failed"
                add_error "service_installation" "Failed to enable $service"
            fi
            
            # Restart service (handles already-running case)
            if sudo systemctl restart "$service"; then
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

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    DEPLOYMENT SUMMARY                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$OVERALL_STATUS" == "success" ]; then
    log "✅ Deployment completed successfully in ${TOTAL_DURATION}s"
elif [ "$OVERALL_STATUS" == "warning" ]; then
    warn "⚠️  Deployment completed with warnings in ${TOTAL_DURATION}s"
else
    err "❌ Deployment failed in ${TOTAL_DURATION}s"
fi

echo ""
log "Deployment report: $REPORT_FILE"
echo ""

# Print validation summary
echo -e "${CYAN}Validation Summary:${NC}"
for phase in "${PHASES[@]}"; do
    status="${PHASE_STATUS[$phase]}"
    duration="${PHASE_DURATION[$phase]:-0}"
    
    if [ "$status" == "success" ]; then
        echo -e "  ${GREEN}✓${NC} $phase (${duration}s)"
    elif [ "$status" == "warning" ]; then
        echo -e "  ${YELLOW}⚠${NC} $phase (${duration}s)"
    elif [ "$status" == "failed" ]; then
        echo -e "  ${RED}✗${NC} $phase (${duration}s)"
    else
        echo -e "  ${YELLOW}○${NC} $phase (skipped)"
    fi
done

echo ""

if [ "$OVERALL_STATUS" != "success" ]; then
    err "Review the deployment report for details: $REPORT_FILE"
    exit 1
fi

log "Services are running. Check status with: sudo systemctl status sieger-inspection sieger-api"
log "View logs with: journalctl -u sieger-inspection -f"

exit 0
