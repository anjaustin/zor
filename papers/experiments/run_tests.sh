#!/bin/bash
# =============================================================================
# ZIT-1 Test Suite Runner
# =============================================================================
# Usage: ./run_tests.sh [--quick|--full|--stress] [-v]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    ZIT-1 TEST SUITE                               ║"
echo "║              Homeo-Adaptive Topological Computation               ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

echo -e "${YELLOW}Compiling test harness...${NC}"

# Compile test harness
nvcc -O3 -o test_harness test_harness.cu 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful.${NC}"
echo ""

# Run tests with passed arguments
./test_harness "$@"
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ALL TESTS PASSED                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    SOME TESTS FAILED                              ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════════════╝${NC}"
fi

exit $TEST_RESULT
