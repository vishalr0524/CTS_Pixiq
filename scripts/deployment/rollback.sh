#!/usr/bin/env bash
# ============================================================================
# Rollback Script — pixIQ Deployment
# ============================================================================
#
# Rolls back a failed deployment to the previous working state.
#
# Usage:
#     ./scripts/deployment/rollback.sh
#     ./scripts/deploy.sh --rollback
#
# ============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[Rollback]${NC} $*"; }
warn() { echo -e "${YELLOW}[Rollback]${NC} $*"; }
err()  { echo -e "${RED}[Rollback]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROLLBACK_REPORT="/opt/sieger/rollback_report_$TIMESTAMP.json"

echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║                    DEPLOYMENT ROLLBACK                         ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_ROOT"

# ── Stop Services ────────────────────────────────────────────────────────────
log "Stopping services..."

for service in sieger-inspection sieger-api; do
    if systemctl is-active --quiet "$service"; then
        sudo systemctl stop "$service"
        log "$service stopped"
    else
        log "$service already stopped"
    fi
done

# ── Restore Previous Git Commit ──────────────────────────────────────────────
log "Checking git history..."

if [ -d .git ]; then
    CURRENT_COMMIT=$(git rev-parse --short HEAD)
    log "Current commit: $CURRENT_COMMIT"
    
    # Check if there are previous commits
    if git rev-parse HEAD~1 &> /dev/null; then
        PREVIOUS_COMMIT=$(git rev-parse --short HEAD~1)
        
        warn "Rolling back to previous commit: $PREVIOUS_COMMIT"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git reset --hard HEAD~1
            log "Git rolled back to: $PREVIOUS_COMMIT"
        else
            warn "Git rollback cancelled"
        fi
    else
        warn "No previous commit found to rollback to"
    fi
else
    warn "Not a git repository"
fi

# ── Restore Config Backup ────────────────────────────────────────────────────
log "Checking for config backups..."

CONFIG_FILE="src/config.json"
BACKUP_FILES=($(ls -t "${CONFIG_FILE}.backup."* 2>/dev/null || true))

if [ ${#BACKUP_FILES[@]} -gt 0 ]; then
    LATEST_BACKUP="${BACKUP_FILES[0]}"
    warn "Found config backup: $LATEST_BACKUP"
    read -p "Restore this backup? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$LATEST_BACKUP" "$CONFIG_FILE"
        log "Config restored from: $LATEST_BACKUP"
    else
        warn "Config restore cancelled"
    fi
else
    log "No config backups found"
fi

# ── Restart Services ─────────────────────────────────────────────────────────
log "Restarting services..."

for service in sieger-inspection sieger-api; do
    if sudo systemctl start "$service"; then
        log "$service started"
    else
        err "$service failed to start"
    fi
done

# ── Generate Rollback Report ─────────────────────────────────────────────────
cat > "$ROLLBACK_REPORT" <<EOF
{
  "rollback_id": "rollback_$TIMESTAMP",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "actions": [
    "Services stopped",
    "Git rollback executed",
    "Config restore attempted",
    "Services restarted"
  ],
  "status": "completed"
}
EOF

log "Rollback report saved: $ROLLBACK_REPORT"

echo ""
warn "Rollback completed. Please verify system status:"
echo "  sudo systemctl status sieger-inspection sieger-api"
echo "  journalctl -u sieger-inspection -n 50"
echo ""

exit 0
