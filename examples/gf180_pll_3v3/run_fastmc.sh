#!/usr/bin/env bash
# Run FastMC on the GF180 PLL testcase.
#
# Before running:
#   1. Build the patched ngspice:  cd ../../ && ./build_ngspice.sh --prefix $HOME/.local
#   2. Set the two paths below.
#
# Usage:
#   chmod +x run_fastmc.sh
#   ./run_fastmc.sh

set -euo pipefail

# ── USER SETTINGS ─────────────────────────────────────────────────────────────

# Path to the GF180 PDK root (directory that contains models/ngspice/)
PDK_ROOT="/home/users/academyadmin/Documents/trial/gf180mcu_fd_pr"

# Path to the directory containing TOP_pex_extracted.spice
NETLIST_DIR="/home/users/academyadmin/Documents/trial/gf180_pll_3v3/shipped_pex_io"

# Path to the patched ngspice binary (built by build_ngspice.sh)
NGSPICE_BIN="${HOME}/.local/bin/ngspice"

# Parallel ngspice processes
NGSPICE_NP=4

# Training seeds (simulated with full circuit solver)
N_TRAIN=200

# Virtual pool size (parameter draws only, no circuit solve)
N_POOL=100000

# Output directory for this run
OUTDIR="$(pwd)/fastmc_output"

# ── DERIVED PATHS ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTMC_DIR="${SCRIPT_DIR}/../.."
TESTBENCH="${SCRIPT_DIR}/pll_tb.sp"

# ── PATCH TESTBENCH PATHS ─────────────────────────────────────────────────────
# Write a resolved copy of pll_tb.sp with the actual PDK and netlist paths.

RESOLVED_TB="${OUTDIR}/pll_tb_resolved.sp"
mkdir -p "${OUTDIR}"
sed \
    -e "s|PDK_ROOT|${PDK_ROOT}|g" \
    -e "s|NETLIST|${NETLIST_DIR}|g" \
    "${TESTBENCH}" > "${RESOLVED_TB}"

echo "=== FastMC — GF180 PLL ==="
echo "    Testbench  : ${RESOLVED_TB}"
echo "    PDK root   : ${PDK_ROOT}"
echo "    Netlist dir: ${NETLIST_DIR}"
echo "    ngspice    : ${NGSPICE_BIN}"
echo "    Training   : ${N_TRAIN} seeds"
echo "    Pool       : ${N_POOL} virtual seeds"
echo "    Output     : ${OUTDIR}"
echo

# ── RUN FASTMC ────────────────────────────────────────────────────────────────

python "${FASTMC_DIR}/fastmc.py" \
    -spice   "${RESOLVED_TB}" \
    -spec    "t_period=20.83e-9" \
    --simulator  ngspice_mc \
    --ngspice-bin "${NGSPICE_BIN}" \
    --ngspice-np  "${NGSPICE_NP}" \
    -n       "${N_TRAIN}" \
    -N       "${N_POOL}" \
    -outdir  "${OUTDIR}"
