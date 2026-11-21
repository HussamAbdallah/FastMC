# FastMC — ngspice Edition

A surrogate-modelling framework for fast Monte Carlo analysis of analog circuits,
using **ngspice** as the simulation backend.

## Overview

FastMC trains a neural-network surrogate on a small set of ngspice Monte Carlo
simulations, then uses active learning to focus simulation effort on the
statistically interesting region of the parameter space. Once trained, the
surrogate can evaluate millions of virtual Monte Carlo seeds in seconds — orders
of magnitude faster than running each seed through the full circuit solver.

This edition uses two new ngspice options (`mc_params_only` and `monte_seed`,
contributed upstream) to separate parameter-matrix generation from circuit
simulation:

| Option | Effect |
|---|---|
| `.options mc_params_only=1` | Run Monte Carlo but skip the circuit solver; output only the parameter draws |
| `.options monte_seed=N` | Start the Monte Carlo random sequence at seed N (reproducible, addressable) |

Together they make it possible to generate a pool of 100 k – 1 M parameter draws
in seconds, then run the full circuit solver only on the small fraction the
surrogate flags as near the failure boundary.

## Repository layout

```
fastmc-ngspice/
├── fastmc.py              # Main CLI entry point
├── experiment.py          # Experiment orchestration (ngspice_mc / ngspice)
├── Simulator.py           # Low-level ngspice runner
├── parser.py              # Output file parser
├── setup.py               # Python package metadata
├── requirements.txt       # Python dependencies
│
├── build_ngspice.sh       # One-command build: patch → compile (no download)
├── mc_params_only_monte_seed.patch   # The patch (applied by build_ngspice.sh)
├── ngspice-47/            # Bundled ngspice 47 source tree (from SourceForge)
├── ngspice_src/           # Modified ngspice 47 source files (patch targets, for reference)
│   ├── src/include/ngspice/tskdefs.h
│   ├── src/include/ngspice/optdefs.h
│   ├── src/spicelib/analysis/cktsopt.c
│   ├── src/spicelib/parser/inp2dot.c
│   ├── src/frontend/inpcom.c
│   ├── src/frontend/numparam/xpressn.c
│   └── src/frontend/runcoms.c
│
├── ngspice_modifications.md  # Detailed description of the patch
└── examples/
    └── gf180_pll_3v3/        # GF180 PLL testcase
```

## 1. Build ngspice (Linux)

The ngspice 47 source tree is bundled in `ngspice-47/` — no download needed.
The build script copies the source, applies the patch, and installs the binary:

```bash
chmod +x build_ngspice.sh
./build_ngspice.sh --prefix $HOME/.local
```

The patched binary lands at `$HOME/.local/bin/ngspice`.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Run FastMC

```bash
python fastmc.py \
  -spice path/to/testbench.sp \
  -spec vout_freq=1e9 \
  --simulator ngspice_mc \
  --ngspice-bin $HOME/.local/bin/ngspice \
  --ngspice-np 4 \
  -outdir results/
```

Key CLI flags:

| Flag | Default | Description |
|---|---|---|
| `-spice` | (required) | SPICE testbench |
| `-spec` | (required) | Performance specs, e.g. `vout=1e9` |
| `--simulator` | `ngspice_mc` | `ngspice_mc` (patched, recommended) or `ngspice` (legacy) |
| `--ngspice-bin` | `ngspice` | Path to ngspice binary |
| `--ngspice-np` | `4` | Parallel ngspice processes |
| `-n` | `200` | Training seeds |
| `-N` | `100000` | Virtual pool size |
| `-outdir` | `./output` | Results directory |

## Workflow

1. **Training** — runs `n` ngspice MC seeds, extracts X (parameters) and
   Y (performance), trains the neural-network surrogate.
2. **Pool generation** — uses `mc_params_only` to generate `N` parameter draws
   without any circuit simulation (seconds for 100 k seeds).
3. **Active learning** — the surrogate scores the pool, selects candidates near
   the failure boundary, re-simulates only those with the full circuit solver,
   and retrains iteratively.
4. **Output** — failure-rate estimate, feature importance, worst-case seeds, and
   audit artifacts.

## Testcase

A GF180 PLL (3.3 V, 8→48 MHz) testcase is included under `examples/gf180_pll_3v3/`.

| File | Purpose |
|---|---|
| `pll_tb.sp` | SPICE testbench (`.measure t_period` for 48 MHz output) |
| `run_fastmc.sh` | One-command FastMC run — set `PDK_ROOT`, `NETLIST_DIR`, `NGSPICE_BIN` at the top |

```bash
cd examples/gf180_pll_3v3
# edit run_fastmc.sh to set your PDK and netlist paths
chmod +x run_fastmc.sh
./run_fastmc.sh
```

## Patch details

See [`ngspice_modifications.md`](ngspice_modifications.md) for a full technical
description of the patch. The patch is proposed for inclusion in upstream ngspice
(discussion: ngspice-devel mailing list, November 2025).
