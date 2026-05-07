#!/usr/bin/env bash
# ============================================================================
# pixIQ System Setup — Jetson Orin NX
# ============================================================================
#
# One-time setup script for a fresh Jetson Orin NX (JetPack 6.x).
# Installs all system dependencies, Basler pylon SDK, Python environment,
# TensorRT prerequisites, and configures the system for production.
#
# Usage:
#     chmod +x scripts/system_setup.sh
#     sudo ./scripts/system_setup.sh
#
# Environment variables (optional):
#     FACTORY_SUBNET=192.168.1.0/24  — factory LAN subnet (default)
#     FACTORY_NIC=eth0               — factory LAN NIC (auto-detected from subnet)
#     SIEGER_USER=sieger             — system user (default)
#     INSTALL_DIR=/opt/sieger        — install directory (default)
#     PYLON_DEB_DIR=/tmp/pylon       — directory containing pylon .deb (default)
#
# Dual NIC layout (Jetson Orin NX has two Ethernet ports):
#     NIC 1 (factory LAN) → 192.168.1.x → cameras, PLC, HMI (jumbo frames)
#     NIC 2 (internet)    → DHCP         → Azure, SSH, apt   (default MTU)
#
# After running this script:
#     1. Assign static IPs to cameras using PylonIpConfigurator
#     2. Edit src/config.json with site-specific values
#     3. Run: uv run python scripts/export_tensorrt.py
#     4. Install systemd services: see docs/14_deployment.md
#
# ============================================================================

set -euo pipefail

# Ensure CUDA tools are in PATH (nvcc may not be in root's PATH by default)
# /usr/local/cuda is the standard CUDA installation path on Linux/Jetson
if [ -d /usr/local/cuda/bin ]; then
    export PATH=/usr/local/cuda/bin:$PATH
fi

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[pixIQ]${NC} $*"; }
warn() { echo -e "${YELLOW}[pixIQ]${NC} $*"; }
err()  { echo -e "${RED}[pixIQ]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

# ── Preflight checks ────────────────────────────────────────────────────────

if [ "$EUID" -ne 0 ]; then
    err "Must be run as root: sudo ./scripts/system_setup.sh"
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    err "This script is for ARM64 (Jetson). Detected: $ARCH"
    exit 1
fi

# Detect JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_INFO=$(cat /etc/nv_tegra_release)
    log "Tegra release: $JETPACK_INFO"
else
    warn "/etc/nv_tegra_release not found — may not be a Jetson device"
fi

# ── Configuration ────────────────────────────────────────────────────────────

SIEGER_USER="${SIEGER_USER:-sieger}"
INSTALL_DIR="${INSTALL_DIR:-/opt/sieger}"
PYLON_DEB_DIR="${PYLON_DEB_DIR:-/tmp/pylon}"
FACTORY_SUBNET="${FACTORY_SUBNET:-192.168.1.0/24}"
FACTORY_NIC="${FACTORY_NIC:-}"  # Auto-detected from FACTORY_SUBNET
PYTHON_VERSION="3.12"

log "Setup configuration:"
log "  User:           $SIEGER_USER"
log "  Install dir:    $INSTALL_DIR"
log "  Factory subnet: $FACTORY_SUBNET"
log "  Factory NIC:    ${FACTORY_NIC:-auto-detect}"
log "  Python:         $PYTHON_VERSION"

# ============================================================================
# 1. System packages
# ============================================================================
step "1/7  System packages"

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    htop \
    nginx \
    sqlite3 \
    libsqlite3-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libhdf5-dev \
    net-tools \
    ethtool \
    iputils-ping \
    network-manager \
    python3-pip \
    python3-libnvinfer \
    python3-libnvinfer-dev

log "Installing jetson-stats (jtop)..."
pip3 install -U jetson-stats

log "System packages installed"

# ============================================================================
# 2. GigE network optimization (dual NIC)
# ============================================================================
step "2/7  GigE network optimization"

# Jetson Orin NX has dual Ethernet ports — physical traffic separation:
#
#   NIC 1 (factory LAN)  → cameras, PLC, HMI    (192.168.1.x, static IP)
#   NIC 2 (internet)     → Azure uploads, SSH    (DHCP / office LAN)
#
# No software traffic prioritization needed — the two networks never
# share a wire. Camera frames cannot be delayed by cloud uploads.
#
# Jumbo frames and buffer tuning apply ONLY to the factory LAN NIC.
# The internet NIC keeps default settings (MTU 1500).

# ── 2a. Kernel buffer tuning (applies to all NICs) ──────────────────────────
SYSCTL_CONF="/etc/sysctl.d/60-gige-vision.conf"
cat > "$SYSCTL_CONF" << 'EOF'
# GigE Vision camera optimization for Basler cameras
# Large receive buffers prevent packet drops during frame transfer
net.core.rmem_max = 16777216
net.core.rmem_default = 16777216

# Increase network backlog for burst traffic from 3 cameras
net.core.netdev_max_backlog = 10000
EOF

sysctl --system > /dev/null 2>&1
log "GigE receive buffer set to 16MB (net.core.rmem_max)"

# ── 2b. Identify factory LAN NIC ────────────────────────────────────────────
# Auto-detect: the NIC with a 192.168.1.x address is the factory LAN NIC.
# Override with FACTORY_NIC env var if auto-detection is wrong.
if [ -z "$FACTORY_NIC" ]; then
    SUBNET_PREFIX=$(echo "$FACTORY_SUBNET" | cut -d'/' -f1 | sed 's/\.[0-9]*$//')
    FACTORY_NIC=$(ip -o -4 addr show | grep "${SUBNET_PREFIX}\." | awk '{print $2}' | head -1 || true)
fi

if [ -n "$FACTORY_NIC" ]; then
    log "Factory LAN NIC detected: $FACTORY_NIC"
else
    warn "Factory LAN NIC not detected (no interface with ${FACTORY_SUBNET} address)"
    warn "Set FACTORY_NIC=ethX manually or configure after network setup"
fi

# ── 2c. Jumbo frames + offload tuning on factory NIC only ────────────────────
# MTU 9000 reduces per-frame packet count by ~6x (1500→9000 bytes per packet).
# Camera GevSCPSPacketSize=8192 needs MTU >= 8228 (payload + headers).
# Must match switch and camera settings — all devices on the factory LAN need jumbo.
#
# Only applied to the factory NIC — the internet NIC keeps MTU 1500.
NM_DISPATCH="/etc/NetworkManager/dispatcher.d/90-pixiq-network"
cat > "$NM_DISPATCH" << DISPATCH_EOF
#!/bin/bash
# pixIQ factory LAN optimization — applies only to the factory NIC.
# Internet NIC is left untouched (standard MTU 1500).

IFACE="\$1"
ACTION="\$2"

if [ "\$ACTION" != "up" ]; then
    exit 0
fi

# Skip loopback and virtual interfaces
case "\$IFACE" in
    lo|docker*|veth*|br-*) exit 0 ;;
esac

# Check if this interface has a factory subnet address
SUBNET_PREFIX="${SUBNET_PREFIX}"
if ip -4 addr show "\$IFACE" | grep -q "${SUBNET_PREFIX}\."; then
    # Factory LAN NIC — apply jumbo frames and low-latency tuning

    # Jumbo frames (MTU 9000)
    ip link set "\$IFACE" mtu 9000 2>/dev/null && \
        logger -t pixiq "Factory NIC \$IFACE: jumbo frames enabled (MTU 9000)" || \
        logger -t pixiq "Factory NIC \$IFACE: jumbo frames not supported — using default MTU"

    # Disable hardware offloading — prevents reordering/coalescing issues
    # with GigE Vision's UDP streaming protocol (GVSP)
    ethtool -K "\$IFACE" gro off gso off tso off 2>/dev/null || true

    # Disable interrupt coalescing for lowest latency (if supported)
    ethtool -C "\$IFACE" rx-usecs 0 2>/dev/null || true

    logger -t pixiq "Factory NIC \$IFACE: GigE optimization applied"
else
    # Internet NIC — leave defaults (MTU 1500, offloading on)
    logger -t pixiq "Internet NIC \$IFACE: keeping default settings"
fi
DISPATCH_EOF
chmod +x "$NM_DISPATCH"
log "Factory LAN NIC optimization configured (jumbo frames, offload disable)"

# Apply to factory NIC now (don't wait for next link up)
if [ -n "$FACTORY_NIC" ]; then
    ip link set "$FACTORY_NIC" mtu 9000 2>/dev/null && \
        log "  MTU 9000 applied to $FACTORY_NIC" || \
        warn "  MTU 9000 failed on $FACTORY_NIC — check switch supports jumbo"
    ethtool -K "$FACTORY_NIC" gro off gso off tso off 2>/dev/null || true
    ethtool -C "$FACTORY_NIC" rx-usecs 0 2>/dev/null || true
fi

# ── 2d. Routing sanity ──────────────────────────────────────────────────────
# Ensure default route goes through internet NIC, not factory NIC.
# Factory NIC should only route 192.168.1.0/24 traffic.
log "Network layout (dual NIC):"
log "  Factory LAN: ${FACTORY_NIC:-<not detected>} → ${FACTORY_SUBNET} (cameras, PLC, HMI)"
INTERNET_NIC=$(ip route show default 2>/dev/null | awk '{print $5}' | head -1)
log "  Internet:    ${INTERNET_NIC:-<not detected>} → default gateway (Azure, SSH, apt)"
if [ -n "$FACTORY_NIC" ] && [ -n "$INTERNET_NIC" ] && [ "$FACTORY_NIC" = "$INTERNET_NIC" ]; then
    warn "Factory NIC and default route use the same interface ($FACTORY_NIC)"
    warn "This means internet traffic shares the wire with cameras."
    warn "Consider setting the default route to the other NIC."
fi

# ============================================================================
# 3. Basler pylon SDK
# ============================================================================
step "3/7  Basler pylon SDK"

# ⚠ Pylon installer URLs are intentionally hardcoded below.
# To update them, edit this file directly.
# URL 1 = pylon arm64 .deb  (pylon_<version>_aarch64.deb)
# URL 2 = pylon USB/TL supplement .deb  (pylon_<version>_aarch64_setup.deb or similar)
PYLON_URL_1="https://dhvani365-my.sharepoint.com/:u:/g/personal/vishal_dhvaniai_com/IQCvXlxfFvEUQJhImEiy1_C3AcRQGf8IJ88u1r2959jfhYQ?download=1"
PYLON_URL_2="https://dhvani365-my.sharepoint.com/:u:/g/personal/vishal_dhvaniai_com/IQCY6dq16artTa7oGtzBp4tvAb-PUom9KoXR0Mw4hQNu2eM?download=1"
PYLON_DEB_1="/tmp/pylon/pylon_arm64_1.deb"
PYLON_DEB_2="/tmp/pylon/pylon_arm64_2.deb"

if [ -d /opt/pylon ]; then
    PYLON_VER=$(/opt/pylon/bin/pylon-config --version 2>/dev/null || echo "installed")
    log "Basler pylon SDK already installed at /opt/pylon (version: $PYLON_VER) — skipping"
else
    log "Basler pylon SDK not found — downloading installer packages..."
    mkdir -p /tmp/pylon

    _download_pylon_pkg() {
        local url="$1"
        local dest="$2"
        local label="$3"
        if [ -f "$dest" ]; then
            log "  $label already downloaded — skipping"
        else
            log "  Downloading $label ..."
            if wget --show-progress -q -O "$dest" "$url"; then
                SIZE=$(du -h "$dest" | awk '{print $1}')
                log "  $label downloaded ($SIZE) ✓"
            else
                err "  Failed to download $label"
                err "  URL: $url"
                err "  Check internet connectivity and re-run this script."
                return 1
            fi
        fi
    }

    _download_pylon_pkg "$PYLON_URL_1" "$PYLON_DEB_1" "pylon package 1 (main deb)" || true
    _download_pylon_pkg "$PYLON_URL_2" "$PYLON_DEB_2" "pylon package 2 (supplement deb)" || true

    # Install whichever .deb files were successfully downloaded
    PYLON_DEBS_FOUND=()
    [ -f "$PYLON_DEB_1" ] && PYLON_DEBS_FOUND+=("$PYLON_DEB_1")
    [ -f "$PYLON_DEB_2" ] && PYLON_DEBS_FOUND+=("$PYLON_DEB_2")

    if [ ${#PYLON_DEBS_FOUND[@]} -eq 0 ]; then
        warn "No pylon .deb files available — Basler cameras will not work."
        warn "Place .deb files in /tmp/pylon/ and re-run, or install manually:"
        warn "  sudo dpkg -i /tmp/pylon/pylon_*.deb && sudo apt-get install -f -y"
    else
        log "Installing pylon SDK packages: ${PYLON_DEBS_FOUND[*]}"
        dpkg -i "${PYLON_DEBS_FOUND[@]}" 2>/dev/null || true
        apt-get install -f -y
        if [ -d /opt/pylon ]; then
            PYLON_VER=$(/opt/pylon/bin/pylon-config --version 2>/dev/null || echo "installed")
            log "Pylon SDK installed successfully: $PYLON_VER ✓"
        else
            warn "dpkg completed but /opt/pylon not found — check package compatibility."
        fi
    fi
fi

# Set pylon environment for current and future shells
PYLON_ENV="/etc/profile.d/pylon.sh"
cat > "$PYLON_ENV" << 'EOF'
# Basler pylon SDK environment
export PYLON_ROOT=/opt/pylon
export LD_LIBRARY_PATH="${PYLON_ROOT}/lib:${LD_LIBRARY_PATH:-}"
EOF
chmod +x "$PYLON_ENV"
source "$PYLON_ENV" 2>/dev/null || true
log "Pylon environment configured in $PYLON_ENV"

# ============================================================================
# 4. TensorRT verification
# ============================================================================
step "4/7  TensorRT verification"

# TensorRT comes pre-installed with JetPack — verify it's available
if python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null; then
    log "TensorRT available (pre-installed with JetPack)"
else
    warn "TensorRT not found — installing from JetPack packages"
    apt-get install -y \
        python3-libnvinfer \
        python3-libnvinfer-dev \
        tensorrt \
        2>/dev/null || warn "TensorRT packages not found in apt — check JetPack installation"
fi

# Verify CUDA
if nvcc --version 2>/dev/null; then
    log "CUDA toolkit available"
else
    warn "nvcc not found — CUDA toolkit may need: sudo apt install cuda-toolkit-*"
fi

# ============================================================================
# 5. Python environment
# ============================================================================
step "5/7  Python environment"

# Install uv (fast Python package manager) to system-wide location
if command -v uv &> /dev/null; then
    log "uv already installed: $(uv --version)"
else
    log "Installing uv..."
    
    # Install to home directory first (default behavior)
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | grep -E "(installing|uv|uvx)" || true
    
    # Move to system-wide location if installed in home
    if [ -f "$HOME/.local/bin/uv" ]; then
        log "Moving uv to /usr/local/bin..."
        mkdir -p /usr/local/bin
        mv "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || {
            # If move fails (permission issue), try copy as fallback
            cp "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || true
        }
        mv "$HOME/.local/bin/uvx" /usr/local/bin/uvx 2>/dev/null || {
            cp "$HOME/.local/bin/uvx" /usr/local/bin/uvx 2>/dev/null || true
        }
        chmod +x /usr/local/bin/uv* 2>/dev/null || true
    fi
    
    # Refresh command cache to find newly moved binary
    hash -r 2>/dev/null || true
    
    # Verify installation
    if command -v uv &> /dev/null; then
        log "uv installed: $(uv --version)"
    else
        warn "uv installation may have issues, proceeding anyway"
    fi
fi

# Create sieger user if it doesn't exist
if id "$SIEGER_USER" &>/dev/null; then
    log "User '$SIEGER_USER' already exists"
else
    log "Creating user '$SIEGER_USER'"
    useradd -m -s /bin/bash "$SIEGER_USER"
    usermod -aG video,render "$SIEGER_USER"
fi

# Ensure user has GPU access
usermod -aG video,render "$SIEGER_USER" 2>/dev/null || true

# Create install directory
mkdir -p "$INSTALL_DIR"
chown "$SIEGER_USER:$SIEGER_USER" "$INSTALL_DIR"

log "Skipping global pypylon setup — uv sync handles pypylon natively inside the Python 3.12 sandbox."

# ============================================================================
# 6. Jetson power & performance
# ============================================================================
step "6/7  Jetson power & performance"

# Set to maximum performance mode (MAXN)
if command -v nvpmodel &> /dev/null; then
    # Check current mode to avoid redundant calls and reboot prompts
    CURRENT_MODE_ID=$(nvpmodel -q 2>/dev/null | grep "NV Power Mode" | grep -o "[0-9]*$" || echo "unknown")
    
    if [[ "$CURRENT_MODE_ID" != "0" ]]; then
        log "Changing Power Mode from $CURRENT_MODE_ID to 0 (MAXN)..."
        # Use 'yes n' to decline immediate reboot if prompted
        yes n | nvpmodel -m 0 2>/dev/null || true
        echo "REBOOT_REQUIRED=true" >> /tmp/pixiq_deploy_flags 2>/dev/null || true
    else
        log "Power mode is already 0 (MAXN)"
    fi
    
    DISPLAY_MODE=$(nvpmodel -q 2>/dev/null | grep "NV Power Mode" | sed 's/.*: //' || echo "MAXN")
    log "Current status: $DISPLAY_MODE"
else
    warn "nvpmodel not found — cannot set power mode"
fi

# Lock clocks to maximum for consistent inference latency
if command -v jetson_clocks &> /dev/null; then
    jetson_clocks 2>/dev/null || true
    log "Clocks locked to maximum (jetson_clocks)"

    # Make jetson_clocks persist across reboots
    CLOCKS_SERVICE="/etc/systemd/system/jetson-clocks.service"
    if [ ! -f "$CLOCKS_SERVICE" ]; then
        cat > "$CLOCKS_SERVICE" << 'EOF'
[Unit]
Description=Lock Jetson clocks to maximum
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload
        systemctl enable jetson-clocks.service
        log "jetson_clocks service enabled (persists across reboots)"
    fi
else
    warn "jetson_clocks not found"
fi

# Disable desktop GUI to free ~500MB RAM (headless operation)
# The HMI runs on a separate all-in-one touchscreen, not on the Jetson
if systemctl is-active --quiet gdm3 2>/dev/null; then
    warn "Desktop GUI (gdm3) is running — disable to free ~500MB RAM:"
    warn "  sudo systemctl set-default multi-user.target"
    warn "  sudo reboot"
fi

# ============================================================================
# 7. Verify installation
# ============================================================================
step "7/7  Verification"

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│                   pixIQ System Setup Summary                 │"
echo "├──────────────────────────────────────────────────────────────┤"

# System
printf "│  %-20s " "Architecture:"
echo -e "${GREEN}$(uname -m)${NC}                                    │"

# CUDA
printf "│  %-20s " "CUDA:"
if nvcc --version 2>/dev/null | grep -q "release"; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}$CUDA_VER${NC}                                         │"
else
    echo -e "${RED}NOT FOUND${NC}                                     │"
fi

# TensorRT
printf "│  %-20s " "TensorRT:"
if python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null; then
    TRT_VER=$(python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null)
    echo -e "│  TensorRT:             ${GREEN}$TRT_VER${NC}                                         │"
else
    echo -e "${RED}NOT FOUND${NC}                                     │"
fi

# Pylon SDK
printf "│  %-20s " "Pylon SDK:"
if [ -d /opt/pylon ]; then
    PYLON_VER=$(/opt/pylon/bin/pylon-config --version 2>/dev/null || echo "installed")
    echo -e "${GREEN}$PYLON_VER${NC}                                     │"
else
    echo -e "${RED}NOT INSTALLED${NC}                                  │"
fi

# uv
printf "│  %-20s " "uv:"
if command -v uv &>/dev/null; then
    echo -e "${GREEN}$(uv --version 2>/dev/null)${NC}                                      │"
else
    echo -e "${RED}NOT FOUND${NC}                                     │"
fi

# GigE buffer
printf "│  %-20s " "GigE rmem_max:"
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null)
if [ "$RMEM" -ge 16777216 ] 2>/dev/null; then
    echo -e "${GREEN}${RMEM} (16MB)${NC}                            │"
else
    echo -e "${YELLOW}${RMEM} (should be 16777216)${NC}              │"
fi

# Factory NIC + jumbo frames
printf "│  %-20s " "Factory NIC:"
if [ -n "$FACTORY_NIC" ]; then
    MTU=$(ip link show "$FACTORY_NIC" 2>/dev/null | grep -oP 'mtu \K[0-9]+')
    if [ "$MTU" -ge 9000 ] 2>/dev/null; then
        echo -e "${GREEN}$FACTORY_NIC MTU $MTU (jumbo)${NC}                    │"
    else
        echo -e "${YELLOW}$FACTORY_NIC MTU $MTU (want 9000)${NC}               │"
    fi
else
    echo -e "${YELLOW}not detected yet${NC}                              │"
fi

# Internet NIC
printf "│  %-20s " "Internet NIC:"
INTERNET_NIC=$(ip route show default 2>/dev/null | awk '{print $5}' | head -1)
if [ -n "$INTERNET_NIC" ]; then
    echo -e "${GREEN}$INTERNET_NIC (default route)${NC}                    │"
else
    echo -e "${YELLOW}no default route${NC}                              │"
fi

# Power mode
printf "│  %-20s " "Power mode:"
if command -v nvpmodel &>/dev/null; then
    MODE=$(nvpmodel -q 2>/dev/null | grep "NV Power Mode" | sed 's/.*: //')
    echo -e "${GREEN}$MODE${NC}                                          │"
else
    echo -e "${YELLOW}unknown${NC}                                       │"
fi

echo "└──────────────────────────────────────────────────────────────┘"

echo ""
log "System setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place pylon .deb in $PYLON_DEB_DIR and re-run if pylon not installed"
echo "  2. Assign static IPs to cameras: /opt/pylon/bin/PylonIpConfigurator"
echo "  3. Clone/deploy code to $INSTALL_DIR"
echo "  4. cd $INSTALL_DIR && uv sync"
echo "  5. Edit src/config.json with site-specific values"
echo "  6. Export TensorRT engines: uv run python scripts/export_tensorrt.py"
echo "  7. Install services: see docs/14_deployment.md section 14.9"
echo ""
