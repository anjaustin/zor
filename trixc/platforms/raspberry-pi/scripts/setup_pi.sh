#!/bin/bash
# ============================================================================
# TRIXC Pi Setup Script
#
# One-time setup for Raspberry Pi.
# Run this once after cloning the repository.
#
# Usage:
#   ./scripts/setup_pi.sh
#   ./scripts/setup_pi.sh --full     # Include all optional features
#   ./scripts/setup_pi.sh --minimal  # Just SDL2
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                      TRIXC Pi Setup                          ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}\"When some wild-eyed, eight-foot-tall maniac grabs your neck,${NC}"
echo -e "  ${YELLOW}taps the back of your favorite head up against the barroom wall,${NC}"
echo -e "  ${YELLOW}and he looks you crooked in the eye and he asks you if ya paid${NC}"
echo -e "  ${YELLOW}your dues, you just stare that big sucker right back in the eye,${NC}"
echo -e "  ${YELLOW}and you remember what ol' Jack Burton always says:${NC}"
echo ""
echo -e "  ${GREEN}Have ya paid your dues, Jack?${NC}"
echo -e "  ${GREEN}Yes sir, the check is in the mail.\"${NC}"
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""

# Parse arguments
INSTALL_GPIO=false
INSTALL_CAMERA=false
INSTALL_AUDIO=false
INSTALL_PYTHON=false

if [[ "$1" == "--full" ]]; then
    INSTALL_GPIO=true
    INSTALL_CAMERA=true
    INSTALL_AUDIO=true
    INSTALL_PYTHON=true
elif [[ "$1" == "--minimal" ]]; then
    # Just SDL2, nothing else
    :
else
    # Interactive mode
    echo "This script will install the required dependencies for TRIXC Pi."
    echo ""
fi

# Check if running on Raspberry Pi
if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(cat /proc/device-tree/model)
    echo -e "${GREEN}Detected: ${MODEL}${NC}"
else
    echo -e "${YELLOW}Warning: Not detected as Raspberry Pi${NC}"
    echo "Continuing anyway (this might work on other Linux systems)"
fi
echo ""

# Update package list
echo -e "${CYAN}[1/5] Updating package list...${NC}"
sudo apt-get update -qq

# Install SDL2 (required)
echo ""
echo -e "${CYAN}[2/5] Installing SDL2 (required)...${NC}"
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev

# Optional: GPIO support
if [[ "$1" != "--minimal" ]]; then
    echo ""
    if [[ "$INSTALL_GPIO" == "true" ]] || [[ -z "$1" ]]; then
        if [[ -z "$1" ]]; then
            read -p "Install GPIO support (pigpio)? [Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                INSTALL_GPIO=true
            fi
        fi

        if [[ "$INSTALL_GPIO" == "true" ]]; then
            echo -e "${CYAN}[3/5] Installing GPIO support (pigpio)...${NC}"
            sudo apt-get install -y pigpio python3-pigpio
            echo -e "${GREEN}GPIO support installed!${NC}"
            echo "  Note: Use 'sudo' when running GPIO examples"
        fi
    fi
fi

# Optional: Camera support
if [[ "$1" != "--minimal" ]]; then
    echo ""
    if [[ "$INSTALL_CAMERA" == "true" ]] || [[ -z "$1" ]]; then
        if [[ -z "$1" ]]; then
            read -p "Install camera support (libcamera)? [Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                INSTALL_CAMERA=true
            fi
        fi

        if [[ "$INSTALL_CAMERA" == "true" ]]; then
            echo -e "${CYAN}[4/5] Installing camera support...${NC}"
            sudo apt-get install -y libcamera-dev libcamera-apps || {
                echo -e "${YELLOW}libcamera not available, trying v4l-utils...${NC}"
                sudo apt-get install -y v4l-utils
            }
            echo -e "${GREEN}Camera support installed!${NC}"
        fi
    fi
fi

# Optional: Python for model conversion
if [[ "$1" != "--minimal" ]]; then
    echo ""
    if [[ "$INSTALL_PYTHON" == "true" ]] || [[ -z "$1" ]]; then
        if [[ -z "$1" ]]; then
            read -p "Install Python for model conversion? [Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                INSTALL_PYTHON=true
            fi
        fi

        if [[ "$INSTALL_PYTHON" == "true" ]]; then
            echo -e "${CYAN}[5/5] Installing Python + ONNX...${NC}"
            sudo apt-get install -y python3-pip python3-numpy
            pip3 install --user onnx || sudo pip3 install onnx
            echo -e "${GREEN}Python + ONNX installed!${NC}"
            echo "  You can now convert ONNX models to C"
        fi
    fi
fi

# Verify installation
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Setup Complete!                           ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""

# Test SDL2
if sdl2-config --version >/dev/null 2>&1; then
    SDL_VERSION=$(sdl2-config --version)
    echo -e "  ${GREEN}✓${NC} SDL2 version: ${SDL_VERSION}"
else
    echo -e "  ${RED}✗${NC} SDL2 not found"
fi

# Test pigpio
if [[ "$INSTALL_GPIO" == "true" ]]; then
    if command -v pigpiod >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} pigpio installed"
    else
        echo -e "  ${YELLOW}!${NC} pigpio may need manual setup"
    fi
fi

# Test Python
if [[ "$INSTALL_PYTHON" == "true" ]]; then
    if python3 -c "import onnx" 2>/dev/null; then
        ONNX_VERSION=$(python3 -c "import onnx; print(onnx.__version__)")
        echo -e "  ${GREEN}✓${NC} ONNX version: ${ONNX_VERSION}"
    else
        echo -e "  ${YELLOW}!${NC} ONNX import failed (may need PATH update)"
    fi
fi

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo ""
echo -e "    ${GREEN}cd platforms/raspberry-pi${NC}"
echo -e "    ${GREEN}make hello${NC}"
echo ""
echo "  This will build and run the Hello XOR example."
echo "  Tap the screen to cycle through inputs!"
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}\"It's all in the reflexes.\"${NC}"
echo ""
