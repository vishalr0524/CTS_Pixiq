#!/usr/bin/env bash
# ============================================================================
# Bootstrap Script — pixIQ Deployment (Stage 1)
# ============================================================================
#
# Minimal bootstrap script for fresh Jetson devices with no git installed.
# This script can be hosted externally (GitHub Gist, Azure Blob, USB drive).
#
# Usage (online):
#     wget https://gist.github.com/dhvani-cv/{hash}/raw/bootstrap.sh
#     bash bootstrap.sh
#
# Usage (offline USB):
#     bash /media/usb/bootstrap.sh
#
# What this script does:
#     1. Installs minimal prerequisites (git, curl, wget, python3-pip)
#     2. Guides GitHub SSH key setup and validation
#     3. Clones the repository to /opt/sieger/
#     4. Automatically launches scripts/deploy.sh (main deployment)
#     5. Generates /tmp/bootstrap_report.json
#
# This script is IDEMPOTENT — safe to re-run after failures.
#
# ============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[Bootstrap]${NC} $*"; }
warn() { echo -e "${YELLOW}[Bootstrap]${NC} $*"; }
err()  { echo -e "${RED}[Bootstrap]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

# ── Configuration ────────────────────────────────────────────────────────────
REPO_URL="git@github.com:dhvani-cv/cone-transport-system-pixiq.git"
REPO_BRANCH="${DEPLOY_BRANCH:-bugfix}"
INSTALL_DIR="${INSTALL_DIR:-/opt/sieger}"
REPO_DIR="$INSTALL_DIR/cone-transport-system-pixiq"
REPORT_FILE="/tmp/bootstrap_report.json"

# JSON report structure
declare -A BOOTSTRAP_CHECKS

# ── Helper Functions ─────────────────────────────────────────────────────────
add_check() {
    local name=$1
    local status=$2
    local details=${3:-""}
    BOOTSTRAP_CHECKS["$name"]="$status|$details"
}

generate_report() {
    local overall_status=$1
    local start_time=$2
    local end_time=$3
    
    cat > "$REPORT_FILE" <<EOF
{
  "phase": "bootstrap",
  "status": "$overall_status",
  "timestamp_start": "$start_time",
  "timestamp_end": "$end_time",
  "duration_seconds": $((end_time - start_time)),
  "checks": [
EOF
    
    local first=true
    for check_name in "${!BOOTSTRAP_CHECKS[@]}"; do
        IFS='|' read -r status details <<< "${BOOTSTRAP_CHECKS[$check_name]}"
        [[ "$first" == false ]] && echo "," >> "$REPORT_FILE"
        first=false
        cat >> "$REPORT_FILE" <<EOF
    {
      "name": "$check_name",
      "status": "$status",
      "details": "$details"
    }
EOF
    done
    
    cat >> "$REPORT_FILE" <<EOF
  ]
}
EOF
    
    log "Bootstrap report saved to: $REPORT_FILE"
}

# ── Start Bootstrap ──────────────────────────────────────────────────────────
START_TIME=$(date +%s)

log "pixIQ Bootstrap — Stage 1 Deployment"
log "Repository: $REPO_URL (branch: $REPO_BRANCH)"
log "Install directory: $INSTALL_DIR"
echo ""

# ── Step 1: Check if running as root ─────────────────────────────────────────
step "Step 1: Checking privileges"

if [ "$EUID" -ne 0 ]; then
    warn "Not running as root. Some operations may require sudo."
    SUDO="sudo"
else
    log "Running as root"
    SUDO=""
fi

# ── Step 2: Install minimal prerequisites ───────────────────────────────────
step "Step 2: Installing prerequisites"

# Check and install git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    log "git already installed: $GIT_VERSION"
    add_check "git_installed" "success" "version: $GIT_VERSION"
else
    log "Installing git..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y git
    GIT_VERSION=$(git --version | awk '{print $3}')
    log "git installed: $GIT_VERSION"
    add_check "git_installed" "success" "version: $GIT_VERSION"
fi

# Check and install curl
if command -v curl &> /dev/null; then
    log "curl already installed"
    add_check "curl_installed" "success" "$(curl --version | head -n1)"
else
    log "Installing curl..."
    $SUDO apt-get install -y curl
    add_check "curl_installed" "success" "installed"
fi

# Check and install wget
if command -v wget &> /dev/null; then
    log "wget already installed"
    add_check "wget_installed" "success" "$(wget --version | head -n1)"
else
    log "Installing wget..."
    $SUDO apt-get install -y wget
    add_check "wget_installed" "success" "installed"
fi

# Check and install python3-pip
if command -v pip3 &> /dev/null; then
    log "pip3 already installed"
    add_check "pip3_installed" "success" "$(pip3 --version)"
else
    log "Installing python3-pip..."
    $SUDO apt-get install -y python3-pip
    add_check "pip3_installed" "success" "installed"
fi

log "Prerequisites installed successfully"

# ── Step 3: GitHub SSH key setup ─────────────────────────────────────────────
step "Step 3: GitHub SSH authentication"

SSH_KEY_PATH="$HOME/.ssh/id_ed25519"

if [ -f "$SSH_KEY_PATH" ]; then
    log "SSH key already exists: $SSH_KEY_PATH"
    cat "$SSH_KEY_PATH"
    add_check "ssh_key_exists" "success" "$SSH_KEY_PATH"
else
    warn "No SSH key found. Generating new SSH key..."
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "sieger-pixiq-$(hostname)"
    
    log "SSH key generated: $SSH_KEY_PATH"
    add_check "ssh_key_generated" "success" "$SSH_KEY_PATH"
fi

# Start ssh-agent and add key to agent
log "Starting ssh-agent and adding key..."
eval "$(ssh-agent -s)" > /dev/null
ssh-add "$SSH_KEY_PATH" 2>&1 > /dev/null
log "Key added to ssh-agent"
add_check "ssh_agent_setup" "success" "key added to agent"

# Display public key for GitHub
echo ""
warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
warn "ACTION REQUIRED: Add this SSH key to GitHub"
warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "$SSH_KEY_PATH.pub"
echo ""
warn "1. Copy the key above"
warn "2. Go to: https://github.com/settings/ssh/new"
warn "3. Paste the key and save"
echo ""
read -p "Press ENTER after adding the key to GitHub..." </dev/tty

# Validate SSH access to GitHub
log "Validating GitHub SSH access..."
if git ls-remote "$REPO_URL" HEAD &>/dev/null; then
    log "GitHub SSH authentication successful"
    add_check "github_ssh_auth" "success" "authenticated"
else
    err "GitHub SSH authentication failed"
    err "Please ensure your SSH key is added to GitHub"
    add_check "github_ssh_auth" "failed" "authentication failed"
    
    END_TIME=$(date +%s)
    generate_report "failed" "$START_TIME" "$END_TIME"
    exit 1
fi

# ── Step 4: Clone repository ─────────────────────────────────────────────────
step "Step 4: Cloning repository"

# Create install directory
if [ ! -d "$INSTALL_DIR" ]; then
    log "Creating install directory: $INSTALL_DIR"
    $SUDO mkdir -p "$INSTALL_DIR"
    $SUDO chown -R $(whoami):$(whoami) "$INSTALL_DIR"
fi

# Clone or update repository
if [ -d "$REPO_DIR/.git" ]; then
    log "Repository already exists, updating..."
    cd "$REPO_DIR"
    
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "$REPO_BRANCH" ]; then
        warn "Switching from branch '$CURRENT_BRANCH' to '$REPO_BRANCH'"
        git fetch origin
        git checkout "$REPO_BRANCH"
    fi
    
    git pull origin "$REPO_BRANCH"
    COMMIT=$(git rev-parse --short HEAD)
    log "Repository updated to: $COMMIT"
    add_check "repo_updated" "success" "branch: $REPO_BRANCH, commit: $COMMIT"
else
    log "Cloning repository..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    
    cd "$REPO_DIR"
    COMMIT=$(git rev-parse --short HEAD)
    log "Repository cloned successfully"
    add_check "repo_cloned" "success" "branch: $REPO_BRANCH, commit: $COMMIT"
fi

# ── Step 5: Generate bootstrap report ────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

step "Step 5: Bootstrap complete"
generate_report "success" "$START_TIME" "$END_TIME"

log "Bootstrap completed in ${DURATION}s"
log "Repository ready at: $REPO_DIR"

# ── Step 6: Launch main deployment script ────────────────────────────────────
# COMMENTED OUT: Bootstrap finishes here. Run main deployment manually.
#step "Step 6: Launching main deployment"
#
#DEPLOY_SCRIPT="$REPO_DIR/scripts/deploy.sh"
#
#if [ -f "$DEPLOY_SCRIPT" ]; then
#    log "Starting main deployment script..."
#    echo ""
#    
#    # Make script executable
#    chmod +x "$DEPLOY_SCRIPT"
#    
#    # Launch main deployment
#    "$DEPLOY_SCRIPT"
#else
#    err "Main deployment script not found: $DEPLOY_SCRIPT"
#    err "Repository may be incomplete. Please check manually."
#    exit 1
#fi

# ── Bootstrap Complete ───────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   BOOTSTRAP COMPLETED                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Next step: Run the main deployment script manually"
echo ""
echo "  cd $REPO_DIR"
echo "  ./scripts/deploy.sh"
echo ""
exit 0