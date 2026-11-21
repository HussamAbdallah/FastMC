#!/usr/bin/env bash
# Build ngspice 47 with the mc_params_only / monte_seed patch.
# The ngspice-47 source tree is bundled in this repo — no download needed.
#
# Usage:
#   chmod +x build_ngspice.sh
#   ./build_ngspice.sh [--prefix /usr/local]
#
# After a successful build the patched ngspice binary is at
# $PREFIX/bin/ngspice.  Pass --ngspice-bin to fastmc.py to point at it.

set -euo pipefail

PREFIX="/usr/local"
if [[ "${1:-}" == "--prefix" ]]; then PREFIX="$2"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/ngspice-47"
BUILD_DIR="${SCRIPT_DIR}/ngspice_build"

echo "=== FastMC ngspice 47 build ==="
echo "    Source         : ${SRC_DIR}"
echo "    Install prefix : ${PREFIX}"
echo "    Build directory: ${BUILD_DIR}"
echo

# ── 1. Copy source to a clean build directory ────────────────────────────────
echo "[1/4] Preparing build directory..."
rm -rf "${BUILD_DIR}"
cp -r "${SRC_DIR}" "${BUILD_DIR}"

# ── 2. Apply patch ───────────────────────────────────────────────────────────
echo "[2/4] Applying mc_params_only / monte_seed patch..."
patch -p1 -d "${BUILD_DIR}" < "${SCRIPT_DIR}/mc_params_only_monte_seed.patch"
echo "      Patch applied successfully."

# ── 3. Configure & compile ───────────────────────────────────────────────────
echo "[3/4] Configuring and compiling (this takes a few minutes)..."
cd "${BUILD_DIR}"
./configure \
    --prefix="${PREFIX}" \
    --with-x=no \
    --disable-debug \
    --enable-openmp \
    2>&1 | tail -5

make -j"$(nproc)" 2>&1 | tail -5

# ── 4. Install ───────────────────────────────────────────────────────────────
echo "[4/4] Installing to ${PREFIX}..."
make install

echo
echo "=== Build complete ==="
echo "    Binary : ${PREFIX}/bin/ngspice"
echo
echo "Run FastMC with:"
echo "  python fastmc.py -spice tb.sp -spec 'freq=1e9' --ngspice-bin ${PREFIX}/bin/ngspice"
