"""
FastMC — Fast Monte Carlo Surrogate Tool
=========================================
Three-phase surrogate-based pipeline for fast Monte Carlo failure analysis.

  Phase 1  Run N_INITIAL (default 500) random SPICE seeds to build the initial
           training set.

  Phase 2  Train a surrogate on those seeds, then sweep a large native-
           distribution pool (auto-sized for the target sigma level, typically
           ~13.5 M for 5σ) using the surrogate to rank candidates worst-first.

  Phase 3  Simulate the worst predicted candidates in batches of 25.  Stop when
           EITHER the confirmed-failure count reaches the target (--n-failures,
           or auto-computed from --sigma/--cl/--n-err) OR the total simulation
           budget (-n) is exhausted.

Stopping criterion — choose one:
    --n-failures N     Stop as soon as N confirmed failures are found.
                       Takes priority over the auto-computed Poisson target.
    -n N               Hard cap on total SPICE simulations (Phase 1 + Phase 3).
                       Active learning stops when this budget is reached regardless
                       of how many failures have been confirmed (default: 1000).
    Both conditions are always active; whichever fires first wins.

Usage (fresh run):
    python fastmc.py -spice testbench.sp --sigma 5 --n-failures 8
    python fastmc.py -spice testbench.sp --sigma 5 -n 1000 --n-initial 100
    python fastmc.py -spice testbench.sp --save-model ./my_surrogate

Usage (resume from saved surrogate):
    python fastmc.py -spice testbench.sp --load-model ./my_surrogate

Key options:
    -spice             SPICE testbench path (required for simulation)
    -n                 Total simulation budget, Phase 1 + Phase 3 (default: 1000)
    --n-initial        Phase 1 seed count (default: 100)
    --n-failures       Explicit failure target; overrides Poisson auto-compute
    --verify-mc        Brute-force MC verification: run N seeds, compare surrogate top-K vs actual
    --verify-k         Number of worst seeds to compare in verification (default: 50)
    --verify-seeds     Override auto-computed seed count for --verify-mc
    --sigma            Target sigma level for analysis (default: 3.0)
    --cl               Statistical confidence level (default: 0.95)
    --n-err            Allowed number of errors for confidence calc (default: 3)
    --simulator        ngspice_mc (default) or ngspice
    --save-model DIR   Save surrogate after training
    --load-model DIR   Load saved surrogate, skip simulation
    --workdir DIR      Temp directory (default: ./temp_dir)
    --outdir DIR       Output directory for CSVs & plots (default: ./output)
    --drop-specs       Comma-separated spec names to exclude
    --show-plots       Open interactive plots instead of saving PNG
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time

# heavyweight imports are moved inside run_pipeline to ensure fast CLI help


# ===========================================================================
# CLI Definition
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="fastmc",
        description="FastMC: A High-Performance Surrogate Tool for Fast Monte Carlo Simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-spice",  metavar="TESTBENCH",
                   help="Path to the SPICE testbench (.sp / .txt) file (required for simulation/training)")
    p.add_argument("-n",      type=int, default=1000, metavar="SEEDS",
                   help="Hard cap on total SPICE simulations (Phase 1 + Phase 3 active learning). "
                        "Active learning stops when this budget is exhausted regardless of "
                        "confirmed failures. (default: 1000)")

    # Viewing previously saved binary plots
    p.add_argument("--view-results", metavar="FILE",
                   help="Open an interactively viewable binary (.pkl) plots file")

    # Model persistence
    model_grp = p.add_mutually_exclusive_group()
    model_grp.add_argument("--save-model", metavar="DIR",
                   help="Save trained surrogate + preprocessors to DIR after training")
    model_grp.add_argument("--load-model", metavar="DIR",
                   help="Load a previously saved surrogate from DIR, skipping training")

    p.add_argument("--sigma", type=float, default=3.0, metavar="S",
                   help="Target sigma level for verification (default: 3.0)")
    p.add_argument("--cl", type=float, default=0.95, metavar="C",
                   help="Confidence level requirement (default: 0.95)")
    p.add_argument("--n-err", type=int, default=3, metavar="E",
                   help="Allowed number of errors (N_err) (default: 3)")
    p.add_argument("--virtual-samples", type=int, default=None, metavar="N",
                   help="Override auto-computed virtual samples count")
    p.add_argument("--native-pool-size", type=int, default=None, metavar="N",
                   help="Override auto-computed active-learning native candidate pool size "
                        "(default: auto-computed from sigma/confidence-level, no stretch).")
    p.add_argument("--pool-size", type=int, default=2000, metavar="P",
                   help="Size of the unlabeled pool for active learning (default: 2000)")
    p.add_argument("--rerank-every", type=int, default=5, metavar="N",
                   help="(legacy, unused in lasso+rf mode) Re-rank parameter importance every "
                        "N rounds in the PCA active-learning branch (default: 5).")
    p.add_argument("--audit-pool", type=int, default=0, metavar="N",
                   help="Size of a true SPICE .op pool to generate for the Industry Standard Auditor (default: 0)")
    p.add_argument("--workdir",  default="./temp_dir", metavar="DIR",
                   help="Temp directory for ngspice runs (default: ./temp_dir)")
    p.add_argument("--outdir",   default="./output",   metavar="DIR",
                   help="Output directory for CSVs & plots (default: ./output)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--widening", type=float, default=1.30,
                   help="Widening factor for virtual sample sigma to fight model shrinkage (default: 1.30)")
    p.add_argument("--drop-specs", default="lock_voltage,id,param,value", metavar="SPECS",
                   help='Comma-separated spec names to drop from training (default: "lock_voltage,id,param,value")')
    p.add_argument("--show-specs", metavar="SPECS",
                   help="Comma-separated spec names to display in plots/table (default: show all)")
    p.add_argument("--export-csv", metavar="FILE",
                   help="Export the 10,000 virtual samples (Y_virtual) to a CSV file")
    p.add_argument("--show-plots", action="store_true",
                   help="Open interactive plot window (default: save PNG)")
    p.add_argument("--transform", type=str, choices=["auto", "none", "sqrt", "log"], default="auto",
                   help="Target transformation to boost tail resolution. 'auto' (default) picks "
                        "none/sqrt/log per spec based on the raw data's skewness; the other "
                        "choices force that mode uniformly across all specs.")
    p.add_argument("--simulator", type=str, choices=["ngspice", "ngspice_mc"], default="ngspice_mc",
                   help="Simulation backend to use: ngspice_mc (default) or ngspice (legacy).")
    p.add_argument("--ngspice-bin", type=str, default="ngspice", metavar="PATH",
                   help="Path to the ngspice executable (default: ngspice). Use this to point at a "
                        "custom build when using --simulator ngspice or ngspice_mc.")
    p.add_argument("--ngspice-np", type=int, default=4, metavar="N",
                   help="Max parallel ngspice processes for ngspice/ngspice_mc training (default: 4).")
    p.add_argument("--monte-seed", type=int, default=None, metavar="K",
                   help="Run a single full-circuit simulation with .options monte_seed=K and print "
                        "its .measure results. Useful for parallel external execution: launch N "
                        "processes each with a different --monte-seed value. "
                        "Requires --simulator ngspice_mc and -spice.")
    p.add_argument("--feature-selection",
                   type=str, choices=["auto", "pca", "lasso", "rf", "lasso+rf"],
                   default="auto",
                   help="Dimensionality reduction strategy. 'auto' picks lasso+rf when "
                        "features>100, pca otherwise. (default: auto)")
    p.add_argument("--top-k", type=int, default=30, metavar="K",
                   help="Max physical features to keep in lasso/rf/lasso+rf modes (default: 30)")
    p.add_argument("--lasso-alpha", type=float, default=None, metavar="A",
                   help="Lasso regularisation alpha. Default: auto (cross-validated LassoCV)")
    p.add_argument("--n-initial", type=int, default=100, metavar="N",
                   help="Number of uniformly-random SPICE seeds to run in Phase 1 to build "
                        "the initial training set (default: 100). Must be <= -n.")
    p.add_argument("--batch-size", type=int, default=10, metavar="K",
                   help="Number of candidates to simulate per active-learning round (default: 10). "
                        "Smaller = faster surrogate self-correction; larger = fewer surrogate retrains.")
    p.add_argument("--n-failures", type=int, default=None, metavar="N",
                   help="Stop active learning as soon as N confirmed failures are found. "
                        "Overrides the auto-computed Poisson target derived from --sigma/--cl/--n-err. "
                        "Both --n-failures and -n are active simultaneously; whichever fires first wins.")
    p.add_argument("--exhaustive", action="store_true",
                   help="Keep running active learning until the surrogate stops finding new failures. "
                        "Stops after --dry-rounds consecutive rounds with 0 new confirmed failures. "
                        "Ignores -n budget (runs until dry). Use to extract ALL failures in the circuit.")
    p.add_argument("--dry-rounds", type=int, default=5, metavar="K",
                   help="Number of consecutive failure-free rounds before --exhaustive stops (default: 5).")
    p.add_argument("--continue", dest="continue_run", action="store_true",
                   help="Continue from the largest existing checkpoint in the output directory. "
                        "-n becomes the number of ADDITIONAL seeds to simulate on top of the loaded data. "
                        "Combines naturally with --exhaustive to keep searching after a previous run.")
    p.add_argument("--verify-mc", action="store_true",
                   help="Run a brute-force Monte Carlo verification pass after training. "
                        "Simulates --verify-seeds seeds (auto-computed for sigma/CL by default), "
                        "scores them with the trained surrogate, and compares the surrogate's "
                        "predicted worst-K against the actual worst-K for Precision@K and "
                        "Spearman rank correlation. Requires -spice to be specified.")
    p.add_argument("--verify-k", type=int, default=50, metavar="K",
                   help="Number of worst seeds to compare in --verify-mc (default: 50). "
                        "Overlap@K = |actual_worst_K ∩ surrogate_worst_K| / K.")
    p.add_argument("--verify-seeds", type=int, default=None, metavar="N",
                   help="Override the auto-computed seed count for --verify-mc. "
                        "Default: N_NATIVE_POOL_AUTO (same formula as active-learning pool, "
                        "e.g. ~120k for sigma=4 or ~13.5M for sigma=5 at default CL/n-err).")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the interactive confirmation prompt before Phase 2/3 active "
                        "learning (for CI/batch use).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned simulation budget and exit without running anything.")
    return p


# ===========================================================================
# Core Pipeline
# ===========================================================================

def run_pipeline(args):
    if getattr(args, "view_results", None):
        import matplotlib
        # Support both desktop GUIs and headless Linux server environments
        if os.environ.get("DISPLAY") or os.name == "nt":
            try:
                matplotlib.use("TkAgg")
            except Exception:
                matplotlib.use("Agg")
        else:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import joblib
        import numpy as np
        
        target_res = os.path.abspath(args.view_results)
        if not os.path.isfile(target_res):
            sys.exit(f"[FastMC] ERROR: Results file '{target_res}' not found.")

        print(f"[FastMC] Loading result data from {target_res}...", flush=True)
        try:
            # Results bundles are not covered by manifest (they are output artefacts,
            # not surrogate models); load is safe because the file is selected by the
            # user explicitly via --view-results and is validated structurally below.
            data = joblib.load(target_res)
            if not isinstance(data, dict) or "Y_virtual" not in data or "spec_names" not in data:
                raise ValueError("Payload missing required FastMC metadata keys ('Y_virtual', 'spec_names').")
        except Exception as e:
            sys.exit(f"[FastMC Security Error] Failed to load results bundle safely: {e}")
        
        # Apply filter if requested via CLI
        SHOW_SPECS = [s.strip() for s in args.show_specs.split(",") if s.strip()] if args.show_specs else []
        
        # Clean internal noise (id, param, value) from the loaded data
        raw_Y = data["Y_virtual"]
        raw_names = data["spec_names"]
        
        DROP_LIST = ["id", "param", "value", "lock_voltage"]
        valid_indices = [i for i, name in enumerate(raw_names) if name.lower() not in DROP_LIST]
        
        Y_virtual = raw_Y[:, valid_indices]
        spec_names = [raw_names[i] for i in valid_indices]
        
        v_mean = np.mean(Y_virtual, axis=0)
        v_std  = np.std(Y_virtual,  axis=0)
        
        s_low_title  = f"{args.sigma}s-low"
        s_high_title = f"{args.sigma}s-high"
        print(f"\n  {'Spec':<25} {'Mean':>14} {'Std':>14} {s_low_title:>14} {s_high_title:>14}")
        print("  " + "-" * 79)
        for i, name in enumerate(spec_names):
            low  = v_mean[i] - args.sigma * v_std[i]
            high = v_mean[i] + args.sigma * v_std[i]
            print(f"  {name:<25} {v_mean[i]:>14.4e} {v_std[i]:>14.4e} {low:>14.4e} {high:>14.4e}")
            
        if args.export_csv:
            import pandas as pd
            df_v = pd.DataFrame(data["Y_virtual"], columns=data["spec_names"])
            df_v.to_csv(args.export_csv, index=False)
            print(f"[FastMC] Exported virtual samples to {args.export_csv}", flush=True)

        valid_n = data.get("valid_n", len(data["Y_raw"]))
        all_r2  = data.get("all_r2")

        # Align all data with the cleaned spec_names
        Y_raw = data["Y_raw"][:, valid_indices]
        Y_val_real = data.get("Y_val_real")
        Y_val_pred = data.get("Y_val_pred")
        
        if Y_val_real is not None: Y_val_real = Y_val_real[:, valid_indices]
        if Y_val_pred is not None: Y_val_pred = Y_val_pred[:, valid_indices]
        
        all_r2 = data.get("all_r2")
        if all_r2 is not None:
            all_r2 = [all_r2[i] for i in valid_indices]

        valid_n = data.get("valid_n", len(Y_raw))

        plot_results(
            spec_names, Y_raw, Y_virtual,
            Y_val_real, Y_val_pred,
            all_r2, valid_n, 
            SHOW_SPECS, show_plots=True, outdir=None,
            sigma=args.sigma
        )
        return

    if getattr(args, "spice", None) is None and getattr(args, "load_model", None) is None:
        sys.exit("[FastMC] ERROR: -spice argument is required unless using --view-results or --load-model.")

    # ── Single-seed mode: --monte-seed K ──────────────────────────────────────
    if getattr(args, "monte_seed", None) is not None:
        if args.simulator != "ngspice_mc":
            sys.exit("[FastMC] --monte-seed requires --simulator ngspice_mc.")
        if not args.spice:
            sys.exit("[FastMC] --monte-seed requires -spice.")
        from Simulator import Simulator
        from parser import NgspiceParser
        seed_k = args.monte_seed
        sim = Simulator(os.path.abspath(args.spice), workdir=args.workdir,
                        ngspice_cmd=args.ngspice_bin)
        out_dir = os.path.join(args.workdir, f"monte_seed_{seed_k}")
        print(f"[FastMC] Running monte_seed={seed_k}...", flush=True)
        logfile = sim.run_monte_seed(seed_k, output_dir=out_dir)
        with open(logfile) as f:
            logtext = f.read()
        _, targets = NgspiceParser().parse_log(logtext)
        print(f"[FastMC] monte_seed={seed_k} results:")
        for name, val in targets.items():
            print(f"  {name} = {val:.6e}")
        return

    # --- Heavyweight imports deferred for faster CLI response ---
    print("[FastMC] Initialising core engines (TensorFlow, Scikit-learn)...", flush=True)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # set non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import joblib
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import skew as _skew
    from experiment import Experiment
    from feature_selection import compute_feature_importance

    def _apply_reduction(X_norm_full, pca_obj, idx):
        """Reduce a full-width scaled feature matrix to the surrogate's actual model
        input space: PCA projection if a PCA object is fitted, a fixed top-K column
        subset if supervised feature selection picked one, otherwise unchanged."""
        if pca_obj is not None:
            return pca_obj.transform(X_norm_full)
        if idx is not None:
            return X_norm_full[:, idx]
        return X_norm_full

    def _auto_pick_transform(y_col):
        """Pick none/sqrt/log for one spec column based on its own skewness."""
        if not np.all(y_col > 0):
            return "none"  # log/sqrt undefined or unstable for non-positive data
        raw_sk = abs(_skew(y_col))
        if raw_sk <= 0.75:
            return "none"  # not meaningfully skewed -- don't transform
        log_sk = abs(_skew(np.log10(y_col)))
        sqrt_sk = abs(_skew(np.sqrt(y_col)))
        best_mode, best_sk = "none", raw_sk
        if log_sk < best_sk - 0.3:
            best_mode, best_sk = "log", log_sk
        if sqrt_sk < best_sk - 0.3:
            best_mode, best_sk = "sqrt", sqrt_sk
        return best_mode

    SPICE_TB      = os.path.abspath(args.spice) if args.spice else None
    WORKDIR       = args.workdir
    OUTDIR        = args.outdir
    N_TRAIN_SEEDS = args.n
    RANDOM_STATE  = args.random_state
    WIDENING      = args.widening
    DROP_SPECS    = [s.strip() for s in args.drop_specs.split(",") if s.strip()]
    SHOW_SPECS    = [s.strip() for s in args.show_specs.split(",") if s.strip()] if args.show_specs else []
    SHOW_PLOTS    = args.show_plots
    SAVE_MODEL    = args.save_model
    LOAD_MODEL    = args.load_model
    EXPORT_CSV    = args.export_csv
    SIMULATOR     = args.simulator
    NGSPICE_BIN   = args.ngspice_bin
    NGSPICE_NP    = getattr(args, "ngspice_np", 4)
    FEAT_SEL      = getattr(args, "feature_selection", "auto")
    TOP_K         = getattr(args, "top_k", 30)
    LASSO_ALPHA   = getattr(args, "lasso_alpha", None)
    
    # ---- Auto-compute N_VIRTUAL from sigma level ----
    CL           = args.cl
    N_ERR        = args.n_err
    TARGET_SIGMA = args.sigma
    
    from scipy.stats import norm as _norm, poisson as _poisson
    from scipy.optimize import fsolve as _fsolve
    import numpy as _np

    p_failure_real = 2.0 * _norm.sf(TARGET_SIGMA)
    
    # Calculate effective sigma based on the generator stretch factor
    GEN_STRETCH = max(WIDENING, TARGET_SIGMA / 3.0)
    EFFECTIVE_SIGMA = TARGET_SIGMA / GEN_STRETCH
    p_failure_effective = 2.0 * _norm.sf(EFFECTIVE_SIGMA)

    def _poisson_eq(n_exp):
        return _poisson.cdf(N_ERR, n_exp[0]) - (1.0 - CL)
    
    N_exp_sol = _fsolve(_poisson_eq, x0=[N_ERR + 1.0])[0]
    # Bound it to a sensible minimum to avoid crashing on tiny sigmas
    N_VIRTUAL_AUTO = max(int(_np.ceil(N_exp_sol / p_failure_effective)), 1000)

    N_VIRTUAL = args.virtual_samples if args.virtual_samples is not None else N_VIRTUAL_AUTO

    # Active-learning candidate pool: instead of synthetically stretching the
    # distribution to make rare (Nsigma) events reachable with a small pool -- which
    # turned out to push individual real device parameters far enough from nominal to
    # break circuit function -- draw directly from the TRUE (unstretched) distribution
    # at a scale large enough for genuine Nsigma events to show up on their own. Same
    # Poisson-confidence math as N_VIRTUAL above, but divided by the true tail
    # probability instead of the stretched one (no stretch factor involved at all).
    N_NATIVE_POOL_AUTO = max(int(_np.ceil(N_exp_sol / p_failure_real)), 1000)
    N_NATIVE_POOL = getattr(args, "native_pool_size", None) or N_NATIVE_POOL_AUTO

    if not getattr(args, "view_results", None):
        print(f"[FastMC] Command: {' '.join(sys.argv)}", flush=True)
        print(f"[FastMC] Monte Carlo Setup:")
        print(f"  Target sigma     : {TARGET_SIGMA}s")
        print(f"  Confidence Level : {CL*100:.1f}%")
        print(f"  Allowed Errors   : {N_ERR}")
        print(f"  Real tail prob   : {p_failure_real:.3e}  (1 in {int(1/p_failure_real):,})")
        print(f"  Gen Stretch      : {GEN_STRETCH:.2f}x")
        print(f"  Effective Sigma  : {EFFECTIVE_SIGMA:.2f}s")
        print(f"  Eff. tail prob   : {p_failure_effective:.3e}  (1 in {int(1/p_failure_effective):,})")
        print(f"  N_exp computed   : {N_exp_sol:.4f}")
        if args.virtual_samples is None:
            print(f"  Virtual samples  : {N_VIRTUAL:,}  (auto-computed using Eff. Sigma)", flush=True)
        else:
            print(f"  Virtual samples  : {N_VIRTUAL:,}  (overridden by user)", flush=True)
        if getattr(args, "native_pool_size", None) is None:
            print(f"  Native AL pool   : {N_NATIVE_POOL:,}  (auto-computed, true distribution, no stretch)\n", flush=True)
        else:
            print(f"  Native AL pool   : {N_NATIVE_POOL:,}  (overridden by user)\n", flush=True)


    if SPICE_TB and not os.path.exists(SPICE_TB):
        sys.exit(f"[FastMC] ERROR: SPICE file not found: {SPICE_TB}")

    os.makedirs(OUTDIR, exist_ok=True)

    # ── Audit logging (V6) ────────────────────────────────────────────────────
    _audit_log_path = os.path.join(OUTDIR, "fastmc_audit.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(_audit_log_path, encoding="utf-8"),
            logging.StreamHandler(stream=open(os.devnull, "w")),  # suppress duplicate console
        ],
        force=True,
    )
    _log = logging.getLogger("fastmc")
    _log.info("FastMC started. cmd=%s", " ".join(sys.argv))

    # ── Kill switch (V2) ──────────────────────────────────────────────────────
    _kill_event = threading.Event()

    def _on_kill(sig, frame):
        print("\n[FastMC] Signal received — checkpointing and stopping after this batch.",
              flush=True)
        _log.warning("Kill signal %s received. Setting stop flag.", sig)
        _kill_event.set()

    signal.signal(signal.SIGINT,  _on_kill)
    signal.signal(signal.SIGTERM, _on_kill)

    # ── Manifest helpers for model integrity (V1) ─────────────────────────────
    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_manifest(directory, filenames):
        manifest = {os.path.basename(fn): _sha256(os.path.join(directory, fn))
                    for fn in filenames if os.path.exists(os.path.join(directory, fn))}
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

    def _safe_joblib_load(directory, filename):
        """Load a joblib/pickle file only after verifying its SHA-256 against manifest.json."""
        manifest_path = os.path.join(directory, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"[FastMC Security] manifest.json not found in {directory}. "
                "This model was saved before integrity checking was added, or the "
                "directory was tampered with. Re-save the model with --save-model to "
                "generate a manifest, or pass --yes to skip this check at your own risk."
            )
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        path = os.path.join(directory, filename)
        expected = manifest.get(filename)
        if expected is None:
            raise ValueError(f"[FastMC Security] '{filename}' has no entry in manifest.json.")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"[FastMC Security] Integrity check FAILED for {filename}. "
                "File may have been tampered with. Aborting load."
            )
        return joblib.load(path)

    # ── Dry-run exit (V3) ─────────────────────────────────────────────────────
    if getattr(args, "dry_run", False):
        print(f"[FastMC] Dry-run: would submit up to {N_TRAIN_SEEDS} SPICE simulations "
              f"({getattr(args, 'n_initial', 100)} Phase 1 + remainder via active learning) "
              f"on simulator={SIMULATOR}.")
        _log.info("Dry-run exit.")
        return

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # =========================================================================
    # LOAD MODEL MODE — skip SPICE sims and training entirely
    # =========================================================================
    if LOAD_MODEL:
        print(f"\n[FastMC] Loading saved surrogate from: {LOAD_MODEL}", flush=True)
        if not os.path.isdir(LOAD_MODEL):
            sys.exit(f"[FastMC] ERROR: --load-model directory not found: {LOAD_MODEL}")

        try:
            model      = load_model(os.path.join(LOAD_MODEL, "surrogate.keras"))
            selector   = _safe_joblib_load(LOAD_MODEL, "selector.pkl")
            x_scaler   = _safe_joblib_load(LOAD_MODEL, "x_scaler.pkl")
            pca        = _safe_joblib_load(LOAD_MODEL, "pca.pkl")
            y_scaler   = _safe_joblib_load(LOAD_MODEL, "y_scaler.pkl")
            meta       = _safe_joblib_load(LOAD_MODEL, "meta.pkl")
            if not isinstance(meta, dict) or "spec_names" not in meta or "X_raw" not in meta:
                raise ValueError("Model meta.pkl missing mandatory calibration attributes.")

            feat_sel_path = os.path.join(LOAD_MODEL, "feature_select.pkl")
            if os.path.exists(feat_sel_path):
                feat_select_meta = _safe_joblib_load(LOAD_MODEL, "feature_select.pkl")
                feat_select_idx = feat_select_meta.get("feat_select_idx")
                feature_importance_df = feat_select_meta.get("feature_importance_df")
            else:
                feat_select_idx = None
                feature_importance_df = None
            _log.info("Surrogate loaded from %s (integrity verified).", LOAD_MODEL)
        except Exception as e:
            sys.exit(f"[FastMC Security Error] Failed to safely load surrogate artifacts: {e}")

        spec_names = meta["spec_names"]
        X_raw      = meta["X_raw"]
        Y_raw      = meta["Y_raw"]
        valid_n    = meta["valid_n"]
        X_clean    = selector.transform(X_raw)
        transform_modes = meta.get("transform_modes", [getattr(args, "transform", "none")] * len(spec_names))

        print(f"  Surrogate loaded  : {model.name}")
        print(f"  Specs             : {spec_names}")
        print(f"  Training samples  : {valid_n}")
        if pca is not None:
            print(f"  PCA components    : {pca.n_components_}")
        elif feat_select_idx is not None:
            print(f"  Selected features : {len(feat_select_idx)} (supervised selection)")
        else:
            print(f"  Feature reduction : bypassed (raw features used directly)")

        if feature_importance_df is not None:
            print("\n  Top parameter contributions (from training):")
            print(feature_importance_df.head(15).to_string(index=False))

        # Skip to Step 5 (need dummy history and val arrays for plotting)
        history    = None
        best_epoch = None
        X_val = Y_val = Y_val_pred = Y_val_real = None
        all_r2 = meta.get("all_r2", [None] * len(spec_names))
        Y_val_real = meta.get("Y_val_real")
        Y_val_pred = meta.get("Y_val_pred")
        seeds_array = None  # no per-seed audit data available from a loaded model

        # Jump straight to virtual sample generation (Step 5)
        goto_step5 = True
    else:
        goto_step5 = False

    if not goto_step5:
        # -----------------------------------------------------------------------
        # STEP 1 — TRAINING DATA
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STEP 1: Generating / Loading Training Data", flush=True)
        print("=" * 60, flush=True)

        csv_x = os.path.join(OUTDIR, f"X_train_{N_TRAIN_SEEDS}.csv")
        csv_y = os.path.join(OUTDIR, f"Y_train_{N_TRAIN_SEEDS}.csv")

        # Populated during lasso+rf active loop; consumed by verify-mc failure matching.
        fastmc_audit_records = []

        # --continue: find the largest existing checkpoint and extend from it
        _continue_run = getattr(args, "continue_run", False)
        if _continue_run:
            import glob as _glob
            _existing = sorted(
                _glob.glob(os.path.join(OUTDIR, "X_train_*.csv")),
                key=lambda p: int(os.path.basename(p).replace("X_train_", "").replace(".csv", "") or 0),
                reverse=True,
            )
            if _existing:
                _prev_x = _existing[0]
                _prev_y = _prev_x.replace("X_train_", "Y_train_")
                if os.path.exists(_prev_y):
                    _prev_n = int(os.path.basename(_prev_x).replace("X_train_", "").replace(".csv", ""))
                    N_TRAIN_SEEDS += _prev_n  # -n becomes additional seeds
                    csv_x = os.path.join(OUTDIR, f"X_train_{N_TRAIN_SEEDS}.csv")
                    csv_y = os.path.join(OUTDIR, f"Y_train_{N_TRAIN_SEEDS}.csv")
                    print(f"  [--continue] Loaded checkpoint: {_prev_x} ({_prev_n} seeds). "
                          f"New target: {N_TRAIN_SEEDS} seeds (+{N_TRAIN_SEEDS - _prev_n}).", flush=True)
                    X_train_df = pd.read_csv(_prev_x)
                    Y_train_df = pd.read_csv(_prev_y)
                    # Merge previous audit records so verify_seed_match.py sees all runs
                    import json as _json, glob as _jglob
                    _prev_audits = sorted(_jglob.glob(os.path.join(OUTDIR, "fastmc_audit_*.json")))
                    if _prev_audits:
                        _aj = _prev_audits[-1]  # largest N = most complete
                        try:
                            with open(_aj) as _af:
                                _loaded = _json.load(_af)
                            fastmc_audit_records.extend(_loaded)
                            print(f"  [--continue] Loaded {len(_loaded)} prior audit records "
                                  f"from {os.path.basename(_aj)}", flush=True)
                        except Exception as _e:
                            print(f"  [--continue] WARNING: could not load audit records: {_e}", flush=True)
                else:
                    print(f"  [--continue] WARNING: found {_prev_x} but no matching Y file — starting fresh.",
                          flush=True)
                    _continue_run = False
            else:
                print(f"  [--continue] No existing checkpoint in {OUTDIR} — starting fresh.", flush=True)
                _continue_run = False

        if not _continue_run and os.path.exists(csv_x) and os.path.exists(csv_y):
            print(f"  Loading cached training data from {csv_x}...", flush=True)
            X_train_df = pd.read_csv(csv_x)
            Y_train_df = pd.read_csv(csv_y)
        elif not _continue_run:
            exp = Experiment(SPICE_TB, workdir=WORKDIR, debug=True, simulator=SIMULATOR, ngspice_cmd=NGSPICE_BIN, ngspice_np=NGSPICE_NP)
            N_INITIAL = min(getattr(args, "n_initial", 500), N_TRAIN_SEEDS)

            # Phase 1: run N_INITIAL uniformly-random MC seeds.
            # NOTE: Mahalanobis-distance tail selection from a large pool was benchmarked
            # but found impractical at n_feat=500+: (a) generating all n_feat columns per
            # chunk costs 8-43 minutes for the 13.5M pool, and (b) the L2-norm
            # concentrates tightly around sqrt(n_feat) in high dimensions (concentration
            # of measure), so the top-500 by norm are only ~1.1x more extreme than a
            # plain random draw -- negligible improvement for the surrogate training.
            # Random seeds are nearly as informative and drastically faster; the surrogate
            # trained on them is what finds actual failures in Phase 2/3.
            print(f"  Phase 1: Running {N_INITIAL} uniformly-random SPICE seeds...", flush=True)
            X_train_df, Y_train_df = exp.execute_training(seeds=range(1, N_INITIAL + 1))

            if N_TRAIN_SEEDS > N_INITIAL:
                RERANK_EVERY = max(1, getattr(args, "rerank_every", 5))
                active_budget = N_TRAIN_SEEDS - len(X_train_df)
                print(f"\n  Phase 2: Surrogate-guided active simulation ({active_budget} budget, "
                      f"batch {getattr(args, 'batch_size', 10)}, pool {N_NATIVE_POOL:,})...", flush=True)

                # ── HITL gate (V3) ────────────────────────────────────────────
                if not getattr(args, "yes", False):
                    print(f"\n[FastMC] About to submit up to {active_budget} SPICE simulations "
                          f"via {SIMULATOR}. Use --yes / -y to skip this prompt in CI.",
                          flush=True)
                    try:
                        ans = input("  Proceed with active learning? [y/N] ").strip().lower()
                    except EOFError:
                        ans = "n"
                    if ans != "y":
                        print("[FastMC] Aborted by user before Phase 2.", flush=True)
                        _log.warning("User aborted before Phase 2 active learning.")
                        X_train_df.to_csv(csv_x, index=False)
                        Y_train_df.to_csv(csv_y, index=False)
                        sys.exit(0)
                _log.info("Phase 2/3 active learning started: budget=%d pool=%d simulator=%s",
                          active_budget, N_NATIVE_POOL, SIMULATOR)

                # Synchronise column list and wiring for the shared audit helper.
                cols = exp.active_params
                force_mechanism = ".alter"
                X_train_df = X_train_df.reindex(columns=cols, fill_value=0.0)

                def _resolve_feat_sel_mode(n_features):
                    return FEAT_SEL if FEAT_SEL != "auto" else ("lasso+rf" if n_features >= 100 else "pca")

                ACTIVE_LEARNING_MODE = _resolve_feat_sel_mode(len(cols))
                AL_BATCH = max(1, getattr(args, "batch_size", 10))

                def _execute_audit_round(X_selected_df, k_points):
                    """Shared real-simulation + bookkeeping step used by both active
                    learning modes below: force-simulate X_selected_df for real, verify
                    absorption, align/append into X_train_df/Y_train_df."""
                    nonlocal X_train_df, Y_train_df
                    print(f"  -> Selected {k_points} extreme candidates. Forcing via {force_mechanism}...")

                    audit_dir = os.path.join(WORKDIR, "al_audits")
                    X_new_df, Y_new_df = exp.execute_audit(X_selected_df, audit_dir=audit_dir)

                    if len(X_new_df) > 0 and len(X_selected_df) > 0:
                        common_cols = [c for c in X_selected_df.columns if c in X_new_df.columns]
                        if common_cols and len(X_new_df) == len(X_selected_df):
                            err = np.mean(np.abs(X_new_df[common_cols].values - X_selected_df[common_cols].values))
                            print(f"  -> SPICE Parameter Absorption Check (Mean Abs Error): {err:.4e}")
                            if err > 1e-2:
                                print(f"  [WARNING] SPICE may not be absorbing the {force_mechanism} values properly! Error is high.")
                        elif common_cols:
                            print(f"  [WARNING] Absorption check skipped: audit returned {len(X_new_df)} rows "
                                  f"for {len(X_selected_df)} requested points (row-count mismatch).")

                    min_len = min(len(X_new_df), len(Y_new_df))
                    X_new_df = X_new_df.iloc[:min_len].reset_index(drop=True)
                    Y_new_df = Y_new_df.iloc[:min_len].reset_index(drop=True)

                    target_cols = [c for c in Y_train_df.columns if c != "Seed"]
                    matching_y_cols = [c for c in target_cols if c in Y_new_df.columns]
                    if matching_y_cols:
                        Y_new_df = Y_new_df[matching_y_cols]

                    if "Seed" not in Y_new_df.columns:
                        dummy_seeds = [100000 + len(X_train_df) + i for i in range(len(Y_new_df))]
                        Y_new_df.insert(0, "Seed", dummy_seeds)

                    X_train_df = pd.concat([X_train_df, X_new_df], ignore_index=True)
                    Y_train_df = pd.concat([Y_train_df, Y_new_df], ignore_index=True)
                    print(f"  -> Appended {len(Y_new_df)}/{k_points} newly-simulated real samples "
                          f"(total now {len(X_train_df)}).", flush=True)

                if ACTIVE_LEARNING_MODE == "pca":
                    # (<100 features, or --feature-selection pca explicitly forced.)
                    # Pool is generated once and re-scored each round with the updated surrogate.
                    POOL_SIZE = min(N_NATIVE_POOL, 500_000)

                    print(f"  -> Generating native pool ({POOL_SIZE:,} candidates)...", flush=True)
                    X_pool_full = exp.execute_generation(n_samples=POOL_SIZE)
                    X_pool_full = X_pool_full.reindex(columns=cols, fill_value=0.0)
                    X_pool_raw = X_pool_full.values
                    # Pool is generated once — native distribution doesn't change,
                    # only the surrogate does. Re-score the same candidates each round.
                    _already_selected = set()

                    while len(X_train_df) < N_TRAIN_SEEDS:
                        if _kill_event.is_set():
                            print("[FastMC] Kill switch active — saving checkpoint and stopping.",
                                  flush=True)
                            _log.warning("Kill switch: PCA loop stopped at %d/%d seeds.",
                                         len(X_train_df), N_TRAIN_SEEDS)
                            X_train_df.to_csv(csv_x, index=False)
                            Y_train_df.to_csv(csv_y, index=False)
                            break
                        k_points = min(AL_BATCH, N_TRAIN_SEEDS - len(X_train_df))
                        print(f"\n  [Active Loop] Current Samples: {len(X_train_df)} / {N_TRAIN_SEEDS}")

                        # Score all unselected pool candidates with the current surrogate
                        remaining_pool_idx = np.array([i for i in range(len(X_pool_raw))
                                                        if i not in _already_selected])

                        valid = (X_train_df != 0).any(axis=1) & Y_train_df.notna().all(axis=1)
                        X_subset = X_train_df[valid].reindex(columns=cols, fill_value=0.0).values.astype("float64")
                        Y_subset = Y_train_df[valid].drop(columns=["Seed"] + DROP_SPECS, errors="ignore").values.astype("float64")

                        try:
                            selector = VarianceThreshold(threshold=0.0)
                            X_c = selector.fit_transform(X_subset)
                            if X_c.shape[1] == 0: X_c = X_subset

                            x_scaler = StandardScaler()
                            X_norm = x_scaler.fit_transform(X_c)

                            pca = PCA(n_components=min(0.99, X_norm.shape[0]-1), random_state=RANDOM_STATE)
                            X_pca = pca.fit_transform(X_norm)

                            y_scaler = StandardScaler()
                            Y_norm = y_scaler.fit_transform(Y_subset)
                        except Exception as e:
                            print(f"  [Active Loop Error] Preprocessing failed: {e}")
                            break

                        print(f"  -> Training temporary MLP surrogate for scoring...")
                        model_tmp = Sequential([
                            Dense(64, activation="relu", input_shape=(X_pca.shape[1],)),
                            BatchNormalization(),
                            Dropout(0.2),
                            Dense(32, activation="relu"),
                            BatchNormalization(),
                            Dense(Y_norm.shape[1])
                        ])
                        model_tmp.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="mse")
                        model_tmp.fit(X_pca, Y_norm, epochs=100, batch_size=16, verbose=0)

                        print(f"  -> Predicting {len(remaining_pool_idx)} pool candidates...")
                        pool_clean = selector.transform(X_pool_raw[remaining_pool_idx])
                        pool_norm  = x_scaler.transform(pool_clean)
                        pool_pca_transform = pca.transform(pool_norm)
                        Y_pool_pred_norm = model_tmp.predict(pool_pca_transform, batch_size=2048, verbose=0)

                        distances = np.sum(np.abs(Y_pool_pred_norm), axis=1)
                        worst_local_idx = np.argsort(distances)[-k_points:]
                        worst_global_idx = remaining_pool_idx[worst_local_idx]

                        X_selected_df = X_pool_full.iloc[worst_global_idx]
                        _execute_audit_round(X_selected_df, k_points)

                        # Mark selected indices so they aren't re-picked next round
                        _already_selected.update(worst_global_idx.tolist())

                else:
                    # (>=100 features, lasso/rf/lasso+rf.)
                    # Draw directly from the true distribution at a scale large enough
                    # (N_NATIVE_POOL) that genuine tail events show up on their own.
                    # Generation samples every parameter independently from its native
                    # agauss marginal. Only the scorer reduces to lasso-selected top-K columns.
                    def _spec_cols_only(df):
                        drop_cols = ["Seed"] + DROP_SPECS + ["id", "param", "value"]
                        return df.drop(columns=drop_cols, errors="ignore")

                    def _setup_pool_generation(X_real_df, Y_real_df, valid_mask):
                        """
                        Run once, right after the initial real seeds: fix the native scale used
                        for generation (agauss-aware) and rank which parameters matter (for the
                        scorer's input, not for generation). Also freezes the mean/std reference
                        (from this same initial batch) used to recognize a "confirmed tail-event"
                        for the early-stopping criterion, and the target count of such events.
                        """
                        X_clean_now = X_real_df[valid_mask].reindex(columns=cols, fill_value=0.0).values.astype("float64")

                        pool_selector = VarianceThreshold(threshold=0.0)
                        X_c_now = pool_selector.fit_transform(X_clean_now)
                        if X_c_now.shape[1] == 0: X_c_now = X_clean_now
                        surviving_cols = list(np.array(cols)[pool_selector.get_support()]) if X_c_now.shape[1] > 0 else list(cols)

                        pool_scaler = StandardScaler()
                        pool_scaler.fit(X_c_now)
                        # See earlier agauss-distribution investigation: reported sample
                        # values for agauss/gauss columns are already native mean=0/std=1
                        # in these units -- more accurate than the noisy fitted scale.
                        if exp.all_agauss(surviving_cols):
                            pool_scaler.mean_ = np.zeros_like(pool_scaler.mean_)
                            pool_scaler.scale_ = np.ones_like(pool_scaler.scale_)
                            pool_scaler.var_ = np.ones_like(pool_scaler.var_)
                        X_n_now = pool_scaler.transform(X_c_now)
                        n_feat = len(surviving_cols)

                        Y_clean_now = _spec_cols_only(Y_real_df[valid_mask]).values.astype("float64")
                        spec_names_now = list(_spec_cols_only(Y_real_df).columns)

                        imp_df = compute_feature_importance(
                            X_n_now, Y_clean_now, surviving_cols, spec_names_now,
                            method=ACTIVE_LEARNING_MODE, top_k=TOP_K, lasso_alpha=LASSO_ALPHA,
                            random_state=RANDOM_STATE,
                        )
                        top_k_now = min(TOP_K, n_feat)
                        imp_sorted = imp_df.nlargest(top_k_now, "importance")
                        important_names = imp_sorted["feature"].tolist()
                        important_idx = [surviving_cols.index(n) for n in important_names]

                        # Bound per-chunk memory/compute by the SCORING width (important_idx),
                        # not the raw feature count: every candidate in a chunk is independently
                        # sampled and scored on just those columns; the other n_feat-len(important)
                        # columns are only ever generated for the handful of candidates that
                        # survive selection (see _sweep_pool below), so they never touch the bulk
                        # per-chunk cost. Target roughly 500MB per generated chunk array.
                        n_important = len(important_idx)
                        chunk_size = max(50_000, min(4_000_000, int(500 * 1024 * 1024 / max(n_important, 1) / 4)))

                        print(f"  -> Importance ranked once on {len(X_clean_now)} initial real "
                              f"samples, for scoring only ({n_important}/{n_feat} lasso-selected "
                              f"columns). Generation independently samples every parameter from "
                              f"its own native agauss distribution -- the other {n_feat - n_important} "
                              f"only for candidates that survive selection -- in chunks of "
                              f"{chunk_size:,}.", flush=True)
                        print(f"  -> Top important parameters (name: importance score):", flush=True)
                        for _, row in imp_sorted.head(10).iterrows():
                            print(f"       {row['feature']}: {row['importance']:.4f}", flush=True)
                        if n_important > 10:
                            print(f"       ... and {n_important - 10} more (this ranking is from the "
                                  f"initial {len(X_clean_now)}-sample batch only -- Step 2 will "
                                  f"re-rank on the full final dataset and write feature_importance.csv).",
                                  flush=True)

                        mean_ref = Y_clean_now.mean(axis=0)
                        std_ref = Y_clean_now.std(axis=0)
                        std_ref[std_ref < 1e-30] = 1e-30
                        n_failures_arg = getattr(args, "n_failures", None)
                        if n_failures_arg is not None:
                            failure_target = int(n_failures_arg)
                            print(f"  -> Failure target: {failure_target} confirmed tail-events "
                                  f"(explicit --n-failures override; Poisson auto would be "
                                  f"{int(np.ceil(N_exp_sol))}).", flush=True)
                        else:
                            failure_target = int(np.ceil(N_exp_sol))
                            print(f"  -> Failure target: {failure_target} confirmed tail-events "
                                  f"(auto-computed from sigma={TARGET_SIGMA}/cl={args.cl}/n-err={args.n_err}).",
                                  flush=True)
                        print(f"     Simulation budget: {N_TRAIN_SEEDS} total seeds. "
                              f"Whichever condition fires first stops active learning.", flush=True)

                        print(f"  -> Pre-generating native pool ({N_NATIVE_POOL:,} candidates) — "
                              f"runs once per active-learning phase; each round re-scores the "
                              f"same pool with the updated surrogate...",
                              flush=True)
                        _t0_pool = time.time()
                        _X_pool_gen = exp.execute_generation(n_samples=N_NATIVE_POOL, seed=1)
                        _X_pool_gen = _X_pool_gen.reindex(columns=cols, fill_value=0.0)
                        X_pool_arr = _X_pool_gen.values.astype("float32")
                        print(f"     Pool ready in {time.time()-_t0_pool:.1f}s. "
                              f"Shape: {X_pool_arr.shape} (~{X_pool_arr.nbytes/1e6:.0f} MB).",
                              flush=True)

                        return {
                            "pool_selector": pool_selector, "pool_scaler": pool_scaler,
                            "important_idx": important_idx, "n_feat": n_feat,
                            "chunk_size": chunk_size,
                            "mean_ref": mean_ref, "std_ref": std_ref,
                            "failure_target": failure_target,
                            "X_pool_arr": X_pool_arr,
                            "pool_selected_idx": set(),  # pool positions already simulated
                        }

                    def _sweep_pool(X_real_df, Y_real_df, valid_mask, gen, k_points):
                        """
                        Every round: fit a fresh temp scorer on the current real data (reduced
                        to the fixed important columns, no PCA), then score the remaining
                        (not-yet-selected) candidates from the pre-generated native pool,
                        returning the top k_points by predicted extremity.
                        """
                        important_idx = gen["important_idx"]
                        n_important = len(important_idx)
                        pool_selector = gen["pool_selector"]
                        pool_scaler = gen["pool_scaler"]
                        X_pool_arr = gen["X_pool_arr"]
                        pool_selected_idx = gen["pool_selected_idx"]

                        X_clean_now = X_real_df[valid_mask].reindex(columns=cols, fill_value=0.0).values.astype("float64")
                        X_c_now = pool_selector.transform(X_clean_now)
                        X_n_now = pool_scaler.transform(X_c_now)
                        Y_clean_now = _spec_cols_only(Y_real_df[valid_mask]).values.astype("float64")

                        # Temp scorer: same lasso-selected columns, no PCA.
                        scorer_x_scaler = StandardScaler().fit(X_n_now[:, important_idx])
                        X_scorer_norm = scorer_x_scaler.transform(X_n_now[:, important_idx])
                        scorer_y_scaler = StandardScaler()
                        Y_norm_tmp = scorer_y_scaler.fit_transform(Y_clean_now)

                        print(f"  -> Training temporary MLP surrogate for scoring on "
                              f"{len(X_scorer_norm)} real samples (lasso features)...", flush=True)
                        model_tmp = Sequential([
                            Dense(64, activation="relu", input_shape=(X_scorer_norm.shape[1],)),
                            BatchNormalization(),
                            Dropout(0.2),
                            Dense(32, activation="relu"),
                            BatchNormalization(),
                            Dense(Y_norm_tmp.shape[1])
                        ])
                        model_tmp.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="mse")
                        tmp_history = model_tmp.fit(X_scorer_norm, Y_norm_tmp, epochs=100, batch_size=16, verbose=0)
                        final_loss = tmp_history.history["loss"][-1]
                        print(f"     scorer trained, {len(tmp_history.history['loss'])} epochs, "
                              f"final MSE loss (normalized units) = {final_loss:.4f}", flush=True)

                        # Determine which pool candidates have not yet been selected.
                        all_pool_idx = np.arange(len(X_pool_arr))
                        remaining_mask = np.ones(len(X_pool_arr), dtype=bool)
                        if pool_selected_idx:
                            remaining_mask[list(pool_selected_idx)] = False
                        remaining_idx = all_pool_idx[remaining_mask]

                        print(f"  -> Scoring {len(remaining_idx):,} remaining pool candidates "
                              f"({len(pool_selected_idx)} already selected across all rounds)...",
                              flush=True)

                        # Score in batches of 10k rows to keep peak memory bounded.
                        SCORE_BATCH = 10_000
                        all_scores = np.empty(len(remaining_idx), dtype="float32")
                        sweep_t0 = time.time()
                        n_score_chunks = int(np.ceil(len(remaining_idx) / SCORE_BATCH))
                        for bi in range(0, len(remaining_idx), SCORE_BATCH):
                            idx_batch = remaining_idx[bi: bi + SCORE_BATCH]
                            X_raw = X_pool_arr[idx_batch].astype("float64")
                            X_c   = pool_selector.transform(X_raw)
                            X_n   = pool_scaler.transform(X_c)
                            X_imp = scorer_x_scaler.transform(X_n[:, important_idx])
                            pred  = model_tmp.predict(X_imp, batch_size=4096, verbose=0)
                            all_scores[bi: bi + len(idx_batch)] = np.sum(np.abs(pred), axis=1)
                            if (bi // SCORE_BATCH + 1) % max(1, n_score_chunks // 5) == 0:
                                pct = min(bi + SCORE_BATCH, len(remaining_idx)) / len(remaining_idx) * 100
                                print(f"     scoring {pct:.0f}% ({time.time()-sweep_t0:.1f}s elapsed)",
                                      flush=True)

                        # Pick top k_points by predicted score.
                        keep_n = min(k_points, len(remaining_idx))
                        top_local = np.argpartition(all_scores, -keep_n)[-keep_n:]
                        top_sorted = top_local[np.argsort(all_scores[top_local])[::-1]]
                        selected_pool_positions = remaining_idx[top_sorted]

                        # Record selected positions so future rounds skip them.
                        pool_selected_idx.update(selected_pool_positions.tolist())

                        shortlist_raw = X_pool_arr[selected_pool_positions]
                        shortlist_scores = all_scores[top_sorted].tolist()

                        # Log the pool positions (0-indexed) and corresponding verify-mc seeds.
                        # seed_k = pool_pos + 2  (row 0 of mc_params.mc was generated by seed 2)
                        seed_labels = [p + 2 for p in selected_pool_positions.tolist()]
                        print(f"     Selected {keep_n} pool positions {selected_pool_positions.tolist()} "
                              f"(verify-mc seeds {seed_labels}). "
                              f"Best score={shortlist_scores[0]:.3f}, "
                              f"worst={shortlist_scores[-1]:.3f} "
                              f"({time.time()-sweep_t0:.1f}s total)", flush=True)

                        # Pass actual pool positions as the DataFrame index so execute_audit
                        # can derive the correct seed_k = pool_pos + 2 for each candidate.
                        return pd.DataFrame(shortlist_raw, index=selected_pool_positions.tolist(), columns=cols), shortlist_scores, selected_pool_positions.tolist()

                    valid_init = (X_train_df != 0).any(axis=1) & Y_train_df.notna().all(axis=1)
                    gen = _setup_pool_generation(X_train_df, Y_train_df, valid_init)

                    # Pre-count confirmed failures from the initial Mahalanobis seeds.
                    # These already count toward the stopping criterion (no need to
                    # re-simulate them in active learning to "discover" they're failures).
                    Y_init_specs = _spec_cols_only(Y_train_df[valid_init])
                    if Y_init_specs.shape[1] == len(gen["mean_ref"]):
                        init_devs = np.sum(
                            np.abs((Y_init_specs.values.astype("float64") - gen["mean_ref"]) / gen["std_ref"]),
                            axis=1,
                        )
                        confirmed_failures = int(np.sum(init_devs > TARGET_SIGMA))
                        print(f"  -> {confirmed_failures}/{valid_init.sum()} initial Mahalanobis seeds "
                              f"are already confirmed tail-events (deviation > {TARGET_SIGMA}σ).", flush=True)
                    else:
                        confirmed_failures = 0

                    _confidence_logged = False
                    _exhaustive = getattr(args, "exhaustive", False)
                    _dry_rounds_limit = getattr(args, "dry_rounds", 5)
                    _consecutive_dry = 0
                    _stop_reason = "budget exhausted before target"
                    # In exhaustive mode -n is a minimum; dry-round stopping only
                    # activates once the minimum budget is consumed.
                    while (len(X_train_df) < N_TRAIN_SEEDS or _exhaustive):
                        if _kill_event.is_set():
                            print("[FastMC] Kill switch active — saving checkpoint and stopping.",
                                  flush=True)
                            _log.warning("Kill switch: lasso+rf loop stopped at %d/%d seeds.",
                                         len(X_train_df), N_TRAIN_SEEDS)
                            X_train_df.to_csv(csv_x, index=False)
                            Y_train_df.to_csv(csv_y, index=False)
                            _stop_reason = "kill switch"
                            break
                        # In non-exhaustive mode, stop at budget
                        if not _exhaustive and len(X_train_df) >= N_TRAIN_SEEDS:
                            _stop_reason = ("confidence achieved"
                                            if confirmed_failures >= gen["failure_target"]
                                            else "budget exhausted before target")
                            break
                        k_points = min(AL_BATCH, N_TRAIN_SEEDS - len(X_train_df)) if not _exhaustive else AL_BATCH
                        _conf_str = (f"confidence ACHIEVED at {confirmed_failures}/{gen['failure_target']}"
                                     if confirmed_failures >= gen["failure_target"]
                                     else f"confirmed tail-events: {confirmed_failures}/{gen['failure_target']}")
                        print(f"\n  [Active Loop] Current Samples: {len(X_train_df)} / {N_TRAIN_SEEDS}  "
                              f"({_conf_str})")

                        valid_now = (X_train_df != 0).any(axis=1) & Y_train_df.notna().all(axis=1)
                        X_selected_df, predicted_scores, _round_pool_positions = _sweep_pool(
                            X_train_df, Y_train_df, valid_now, gen, k_points
                        )

                        n_before = len(X_train_df)
                        _execute_audit_round(X_selected_df, k_points)

                        new_rows = _spec_cols_only(Y_train_df.iloc[n_before:])
                        n_new = len(new_rows)
                        if n_new > 0 and new_rows.shape[1] == len(gen["mean_ref"]):
                            deviations = np.sum(
                                np.abs((new_rows.values.astype("float64") - gen["mean_ref"]) / gen["std_ref"]),
                                axis=1,
                            )
                            round_failures = int(np.sum(deviations > TARGET_SIGMA))
                            confirmed_failures += round_failures

                            n_compare = min(n_new, len(predicted_scores))
                            print(f"     Predicted score (from generation) vs. actual deviation "
                                  f"(real simulation) for this round:", flush=True)
                            print(f"     {'#':<4} {'Predicted score':>16} {'Actual deviation':>18} "
                                  f"{'Pool pos':>10}  {'MC seed':>8}  {'Confirmed failure?':>20}", flush=True)
                            for idx in range(n_compare):
                                flag = "YES" if deviations[idx] > TARGET_SIGMA else ""
                                _pp = _round_pool_positions[idx] if idx < len(_round_pool_positions) else -1
                                print(f"     {idx+1:<4} {predicted_scores[idx]:>16.3f} "
                                      f"{deviations[idx]:>18.3f} {_pp:>10}  {_pp+1:>8}  "
                                      f"{flag:>20}", flush=True)
                            print(f"     -> {round_failures} confirmed tail-event(s) this round "
                                  f"(cumulative {confirmed_failures}/{gen['failure_target']})", flush=True)

                            # Record each audit point for later failure matching with verify-mc.
                            for _ri in range(min(n_new, len(_round_pool_positions))):
                                fastmc_audit_records.append({
                                    "pool_position":    _round_pool_positions[_ri],
                                    "verify_mc_seed":   _round_pool_positions[_ri] + 1,
                                    "predicted_score":  predicted_scores[_ri] if _ri < len(predicted_scores) else None,
                                    "fastmc_deviation": float(deviations[_ri]),
                                    "fastmc_failure":   bool(deviations[_ri] > TARGET_SIGMA),
                                })
                            # Dry-round tracking for --exhaustive mode.
                            # Only activates after the minimum -n budget is consumed.
                            if _exhaustive and len(X_train_df) >= N_TRAIN_SEEDS:
                                if round_failures == 0:
                                    _consecutive_dry += 1
                                    print(f"     -> Dry round {_consecutive_dry}/{_dry_rounds_limit} "
                                          f"(no new failures; {confirmed_failures} total so far)", flush=True)
                                    if _consecutive_dry >= _dry_rounds_limit:
                                        _stop_reason = (f"exhaustive: {_dry_rounds_limit} consecutive "
                                                        f"dry rounds — no more failures found")
                                        print(f"\n  [Exhaustive Stop] {_dry_rounds_limit} consecutive rounds "
                                              f"with 0 new failures after {N_TRAIN_SEEDS} minimum seeds. "
                                              f"Total: {confirmed_failures} failures in "
                                              f"{len(X_train_df)} seeds.", flush=True)
                                        break
                                else:
                                    if _consecutive_dry > 0:
                                        print(f"     -> Dry streak reset (was {_consecutive_dry})", flush=True)
                                    _consecutive_dry = 0

                            if (not _confidence_logged
                                    and confirmed_failures >= gen["failure_target"]):
                                _confidence_logged = True
                                extra = (f"Continuing until surrogate is exhausted (--exhaustive)."
                                         if _exhaustive else
                                         f"Continuing to full budget ({N_TRAIN_SEEDS}) "
                                         f"to maximise surrogate accuracy.")
                                print(f"\n  [Statistical Confidence ACHIEVED] {confirmed_failures} confirmed "
                                      f"tail-events reached target {gen['failure_target']} after "
                                      f"{len(X_train_df)} seeds. {extra}", flush=True)

                    if not _stop_reason.startswith("budget") and not _stop_reason.startswith("kill"):
                        if confirmed_failures >= gen["failure_target"] and not _exhaustive:
                            _stop_reason = "confidence achieved"
                    print(f"\n  [Active Learning] Complete: {len(X_train_df)} seeds used, "
                          f"{confirmed_failures}/{gen['failure_target']} confirmed tail-events "
                          f"({_stop_reason}).",
                          flush=True)

            X_train_df.to_csv(csv_x, index=False)
            Y_train_df.to_csv(csv_y, index=False)
            print(f"  Saved Final Labeled dataset to {csv_x}", flush=True)

            # Save audit records so --continue runs and verify_seed_match.py can merge them
            if fastmc_audit_records:
                import json as _json
                _audit_path = os.path.join(OUTDIR, f"fastmc_audit_{N_TRAIN_SEEDS}.json")
                with open(_audit_path, "w") as _af:
                    _json.dump(fastmc_audit_records, _af)
                print(f"  Saved audit records ({len(fastmc_audit_records)}) → {_audit_path}", flush=True)

        # Always drop internal artifacts if they exist (keep "Seed" for now so it can be
        # filtered by the same row masks as everything else, instead of going stale)
        Y_train_df = Y_train_df.drop(columns=DROP_SPECS + ["id", "param", "value"], errors="ignore")

        # Filter failed seeds
        # A seed is valid if it has non-zero features AND all REMAINING specs are positive
        spec_cols_only = Y_train_df.drop(columns=["Seed"], errors="ignore")
        valid = (X_train_df != 0).any(axis=1) & (spec_cols_only > 0).all(axis=1)
        X_train_df = X_train_df[valid]
        Y_train_df = Y_train_df[valid]

        # --- Extract seeds AFTER row filtering, so seeds_array stays row-aligned with
        # Y_raw/X_raw below (extracting it before the `valid` filter silently shifts the
        # seed labels reported by the Step 6 audit relative to which row they actually
        # belong to whenever any row is dropped here). ---
        if "Seed" in Y_train_df.columns:
            seeds_array = Y_train_df["Seed"].values
            Y_train_df = Y_train_df.drop(columns=["Seed"])
        else:
            seeds_array = None

        print(f"\n  Valid samples : {valid.sum()} / {N_TRAIN_SEEDS}")
        print(f"  Features (X) : {X_train_df.shape[1]}")
        
        spec_names = Y_train_df.columns.tolist()
        print(f"  Specs (Y)    : {spec_names}")

        if valid.sum() < 20:
            sys.exit("[FastMC] ERROR: Too few valid samples. Check your netlist.")

        X_raw = X_train_df.values.astype("float64")
        Y_raw = Y_train_df.values.astype("float64")
        print(f"  Valid samples : {valid.sum()} / {N_TRAIN_SEEDS}")
        print(f"  Features (X) : {X_train_df.shape[1]}")
        print(f"  Specs (Y)    : {spec_names}")

        valid_n = len(Y_raw)
        
        # -----------------------------------------------------------------------
        # STEP 2 — PREPROCESSING (VarianceThreshold → StandardScaler → PCA)
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STEP 2: Preprocessing", flush=True)
        print("=" * 60, flush=True)

        selector = VarianceThreshold(threshold=0.0)
        X_clean  = selector.fit_transform(X_raw)
        print(f"  After VarianceThreshold : {X_clean.shape[1]} features (was {X_raw.shape[1]})")

        x_scaler = StandardScaler()
        X_norm   = x_scaler.fit_transform(X_clean)

        feature_names_clean = list(np.array(X_train_df.columns)[selector.get_support()])

        FEAT_SEL_ACTUAL = FEAT_SEL if FEAT_SEL != "auto" else ("lasso+rf" if X_norm.shape[1] >= 100 else "pca")
        feat_select_idx = None
        feature_importance_df = None
        pca = None

        if FEAT_SEL_ACTUAL == "pca":
            if X_norm.shape[1] < 100:
                print(f"  Bypassing PCA (Features={X_norm.shape[1]} < 100) to preserve tail sensitivity.")
                X_pca = X_norm
            else:
                print("  Applying PCA (99% variance threshold)...")
                pca   = PCA(n_components=0.99, random_state=RANDOM_STATE)
                X_pca = pca.fit_transform(X_norm)
                print(f"  After PCA               : {X_pca.shape[1]} components")
        else:
            print(f"  Using supervised feature selection ({FEAT_SEL_ACTUAL}), top_k={TOP_K}...")
            feature_importance_df = compute_feature_importance(
                X_norm, Y_raw, feature_names_clean, spec_names,
                method=FEAT_SEL_ACTUAL, top_k=TOP_K, lasso_alpha=LASSO_ALPHA,
                random_state=RANDOM_STATE,
            )
            top_k_actual = min(TOP_K, X_norm.shape[1])
            top_names = feature_importance_df.nlargest(top_k_actual, "importance")["feature"].tolist()
            feat_select_idx = np.array([feature_names_clean.index(n) for n in top_names])
            X_pca = X_norm[:, feat_select_idx]
            print(f"  Selected {len(feat_select_idx)} / {X_norm.shape[1]} features "
                  f"(original parameter names kept for interpretability).")

            if 'exp' in locals() and hasattr(exp, 'param_distributions') and exp.param_distributions:
                # sigma here is the derived true one-sigma spread (abs_variation /
                # sigma_mult) from agauss(nominal, abs_variation, sigma)
                # declaration -- not abs_variation itself.
                dist_df = pd.DataFrame([
                    {"feature": name, "dist": d.get("dist"), "nominal": d.get("nominal"),
                     "abs_variation": d.get("abs_variation"), "sigma_mult": d.get("sigma_mult"),
                     "sigma": d.get("sigma")}
                    for name, d in exp.param_distributions.items()
                ])
                feature_importance_df = feature_importance_df.merge(dist_df, on="feature", how="left")

            imp_csv = os.path.join(OUTDIR, "feature_importance.csv")
            feature_importance_df.to_csv(imp_csv, index=False)
            print(f"  Saved parameter-contribution ranking to {imp_csv}")
            print("\n  Top parameter contributions:")
            print(feature_importance_df.head(15).to_string(index=False))

        # Multi-Transform Logic
        USE_TF = args.transform
        if USE_TF == "auto":
            transform_modes = [_auto_pick_transform(Y_raw[:, i]) for i in range(Y_raw.shape[1])]
            print("  Auto-selected target transform per spec:")
            for name, mode in zip(spec_names, transform_modes):
                print(f"    {name}: {mode}")
        else:
            transform_modes = [USE_TF] * Y_raw.shape[1]

        log_cols = [i for i, m in enumerate(transform_modes) if m == "log"]
        if log_cols:
            valid_mask = (Y_raw[:, log_cols] > 0).all(axis=1)
            if not valid_mask.all():
                n_bad = len(Y_raw) - valid_mask.sum()
                print(f"  [Warning] Dropped {n_bad} samples with zero/negative Y values on "
                      f"log-transformed spec(s) (sim failures).")
                X_pca = X_pca[valid_mask]
                X_clean = X_clean[valid_mask]
                X_raw = X_raw[valid_mask]
                Y_raw = Y_raw[valid_mask]
                if seeds_array is not None:
                    seeds_array = seeds_array[valid_mask]

        Y_train_ready = np.empty_like(Y_raw)
        for i, mode in enumerate(transform_modes):
            if mode == "sqrt":
                Y_train_ready[:, i] = np.sqrt(Y_raw[:, i])
            elif mode == "log":
                Y_train_ready[:, i] = np.log10(Y_raw[:, i])
            else:
                Y_train_ready[:, i] = Y_raw[:, i]

        y_scaler = StandardScaler()
        # Final safety check before scaling
        if not np.isfinite(Y_train_ready).all():
            print("  [Warning] Non-finite values found in Y after transform. Cleaning...")
            finite_mask = np.isfinite(Y_train_ready).all(axis=1)
            Y_train_ready = Y_train_ready[finite_mask]
            X_pca = X_pca[finite_mask]
            X_clean = X_clean[finite_mask]
            X_raw = X_raw[finite_mask]
            Y_raw = Y_raw[finite_mask]
            if seeds_array is not None:
                seeds_array = seeds_array[finite_mask]

        Y_norm = y_scaler.fit_transform(Y_train_ready)

        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_pca, Y_norm, test_size=0.2, random_state=RANDOM_STATE
        )
        print(f"  Train / Val split       : {len(X_tr)} / {len(X_val)}")

        # -----------------------------------------------------------------------
        # STEP 3 — TRAIN MLP SURROGATE
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STEP 3: Training MLP Surrogate", flush=True)
        print("=" * 60, flush=True)

        from tensorflow.keras.layers import LeakyReLU
        from tensorflow.keras.regularizers import l2
        
        # Smaller, more robust model appropriate for 500 training points
        # This prevents over-regularization and follows physics slopes better
        model = Sequential([
            Dense(64, input_shape=(X_pca.shape[1],)),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(),
            Dense(32),
            LeakyReLU(negative_slope=0.1),
            Dense(Y_tr.shape[1]),
        ], name="FastMC_Surrogate")

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=100,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                               patience=40, min_lr=1e-7, verbose=1),
        ]

        # Compute sample weights: 1.0 for initial (first 100), 10.0 for active learning seeds
        n_initial = 100 
        n_total   = len(X_pca)
        weights = np.ones(n_total)
        weights[n_initial:] = 25.0 # Extreme priority for the tail-hunters
        
        # We need to split weights if we split the data, 
        # but to keep it simple, we can just use the indices.
        # Let's align indices with the split
        train_idx, val_idx = train_test_split(
            np.arange(n_total), test_size=0.2, random_state=RANDOM_STATE
        )
        X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
        Y_tr, Y_val = Y_norm[train_idx], Y_norm[val_idx]
        W_tr = weights[train_idx]

        history = model.fit(
            X_tr, Y_tr,
            sample_weight=W_tr,
            validation_data=(X_val, Y_val),
            epochs=2000, batch_size=32,
            callbacks=callbacks, verbose=0,
        )

        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        print(f"\n  Stopped at epoch {len(history.history['loss'])} (best: {best_epoch})", flush=True)

        # -----------------------------------------------------------------------
        # STEP 4 — VALIDATION
        # -----------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STEP 4: Validation on Hold-Out Set", flush=True)
        print("=" * 60, flush=True)

        Y_val_pred = y_scaler.inverse_transform(model.predict(X_val, verbose=0))
        Y_val_real = y_scaler.inverse_transform(Y_val)

        all_r2 = []
        for i, name in enumerate(spec_names):
            r2  = r2_score(Y_val_real[:, i], Y_val_pred[:, i])
            mse = mean_squared_error(Y_val_real[:, i], Y_val_pred[:, i])
            mae = mean_absolute_error(Y_val_real[:, i], Y_val_pred[:, i])
            all_r2.append(r2)
            tag = "OK " if r2 >= 0.85 else ("~  " if r2 >= 0.6 else "BAD")
            print(f"  [{tag}] {name}: R2={r2:.4f}  MSE={mse:.3e}  MAE={mae:.3e}")

        print(f"\n  Mean R2: {np.mean(all_r2):.4f}", flush=True)

        # -----------------------------------------------------------------------
        # SELF-CONSISTENCY CHECK (Diagnostic)
        # -----------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("DIAGNOSTIC: MLP Range & Accuracy Check (Training Set)")
        # Transform features
        X_fast_norm = x_scaler.transform(X_clean)
        X_fast_pca = _apply_reduction(X_fast_norm, pca, feat_select_idx)

        Y_train_pred_norm = model.predict(X_fast_pca, verbose=0)
        Y_train_pred_inv  = y_scaler.inverse_transform(Y_train_pred_norm)

        # Apply per-spec inverse transforms for diagnostic reporting
        Y_train_pred = np.empty_like(Y_train_pred_inv)
        for i, mode in enumerate(transform_modes):
            if mode == "sqrt":
                Y_train_pred[:, i] = Y_train_pred_inv[:, i]**2
            elif mode == "log":
                Y_train_pred[:, i] = 10**Y_train_pred_inv[:, i]
            else:
                Y_train_pred[:, i] = Y_train_pred_inv[:, i]
        
        from scipy.stats import kurtosis
        print("\n" + "-" * 60)
        print("SPREAD VALIDATION (Inference vs Training)")
        print("-" * 60)
        for i, name in enumerate(spec_names):
            std_train = np.std(Y_raw[:, i])
            std_inf   = np.std(Y_train_pred[:, i])
            ratio     = std_inf / std_train
            kurt      = kurtosis(Y_train_pred[:, i])
            status    = "PASS" if ratio > 0.9 else "FAIL (Clipped Spread)"
            print(f"  {name[:10]:<10} | Std Ratio: {ratio:.2f} | Kurtosis: {kurt:.2f} -> {status}")
        print("-" * 60)
        
        # Table 1: Side-by-side (First 5)
        print(f"{'Spec':<15} | {'SPICE Raw':<12} | {'MLP Pred':<12} | {'Error %':<10}")
        print("-" * 55)
        for i in range(min(5, len(Y_raw))):
            for s_idx, s_name in enumerate(spec_names):
                actual = Y_raw[i, s_idx]
                pred   = Y_train_pred[i, s_idx]
                err    = abs(actual - pred) / (abs(actual) + 1e-20) * 100
                print(f"{s_name[:15]:<15} | {actual:<12.4e} | {pred:<12.4e} | {err:<10.2f}%")
            if len(spec_names) > 1: print("--")
            
        # Table 2: Range check (Spread)
        print("\nRange Comparison (Spread):")
        print(f"{'Spec':<15} | {'SPICE Min/Max':<25} | {'MLP Min/Max':<25}")
        print("-" * 75)
        for i, name in enumerate(spec_names):
            s_min, s_max = np.min(Y_raw[:, i]), np.max(Y_raw[:, i])
            m_min, m_max = np.min(Y_train_pred[:, i]), np.max(Y_train_pred[:, i])
            print(f"{name[:15]:<15} | {s_min:.2e} to {s_max:.2e} | {m_min:.2e} to {m_max:.2e}")

        # -----------------------------------------------------------------------
        # SAVE MODEL (optional)
        # -----------------------------------------------------------------------
        if SAVE_MODEL:
            os.makedirs(SAVE_MODEL, exist_ok=True)
            model.save(os.path.join(SAVE_MODEL, "surrogate.keras"))
            joblib.dump(selector,  os.path.join(SAVE_MODEL, "selector.pkl"))
            joblib.dump(x_scaler,  os.path.join(SAVE_MODEL, "x_scaler.pkl"))
            joblib.dump(pca,       os.path.join(SAVE_MODEL, "pca.pkl"))
            joblib.dump(y_scaler,  os.path.join(SAVE_MODEL, "y_scaler.pkl"))
            if feat_select_idx is not None:
                joblib.dump({
                    "feat_select_idx": feat_select_idx,
                    "feature_names_clean": feature_names_clean,
                    "feature_importance_df": feature_importance_df,
                }, os.path.join(SAVE_MODEL, "feature_select.pkl"))
            _saved_pkls = ["selector.pkl", "x_scaler.pkl", "pca.pkl", "y_scaler.pkl"]
            if feat_select_idx is not None:
                _saved_pkls.append("feature_select.pkl")
            joblib.dump({
                "spec_names":      spec_names,
                "X_raw":           X_raw,
                "X_train_columns": list(X_train_df.columns),
                "Y_raw":           Y_raw,
                "valid_n":         valid_n,
                "all_r2":          all_r2,
                "Y_val_real":      Y_val_real,
                "Y_val_pred":      Y_val_pred,
                "transform_modes": transform_modes,
            }, os.path.join(SAVE_MODEL, "meta.pkl"))
            _saved_pkls.append("meta.pkl")
            _write_manifest(SAVE_MODEL, _saved_pkls)
            _log.info("Surrogate saved to %s with integrity manifest.", SAVE_MODEL)
            print(f"\n  Surrogate saved to: {SAVE_MODEL}", flush=True)
            print(f"  Integrity manifest: {os.path.join(SAVE_MODEL, 'manifest.json')}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 3b — FINAL SURROGATE: SCORE FULL NATIVE POOL (worst predictions)
    # -----------------------------------------------------------------------
    # After the surrogate is trained, apply it to every candidate in the
    # pre-generated native-distribution pool (gen["X_pool_arr"]) and rank by
    # predicted deviation.  The top-K worst predicted candidates are saved to
    # CSV so the user can inspect which corners of the design space the model
    # says are most failure-prone, before the brute-force verify-MC confirms.
    # -----------------------------------------------------------------------
    _pool_scores_csv = os.path.join(OUTDIR, "pool_worst_predictions.csv")
    if ('gen' in locals() and gen is not None and 'X_pool_arr' in gen
            and 'model' in locals() and model is not None):
        import numpy as _np3
        _X_pool_raw = gen["X_pool_arr"]   # shape (N_NATIVE_POOL, n_raw_cols), float32
        _K_WORST    = getattr(args, "verify_k", 50)
        print("\n" + "=" * 60, flush=True)
        print(f"STEP 3b: Final Surrogate — Scoring {len(_X_pool_raw):,} Pool Candidates",
              flush=True)
        print("=" * 60, flush=True)

        _PRED_BATCH = 10_000
        _pool_pred_list = []
        for _bi in range(0, len(_X_pool_raw), _PRED_BATCH):
            _Xb = _X_pool_raw[_bi: _bi + _PRED_BATCH].astype("float64")
            _Xc = selector.transform(_Xb)
            _Xn = x_scaler.transform(_Xc)
            if feat_select_idx is not None:
                _Xp = _Xn[:, feat_select_idx]
            elif pca is not None:
                _Xp = pca.transform(_Xn)
            else:
                _Xp = _Xn
            _pool_pred_list.append(model.predict(_Xp, batch_size=4096, verbose=0))
        _Y_pool_norm = _np3.concatenate(_pool_pred_list, axis=0)

        # Inverse-transform to raw spec space
        _Y_pool_scaled = y_scaler.inverse_transform(_Y_pool_norm)
        _Y_pool_raw = _np3.empty_like(_Y_pool_scaled)
        for _i, _mode in enumerate(transform_modes):
            if _mode == "sqrt":
                _Y_pool_raw[:, _i] = _np3.clip(_Y_pool_scaled[:, _i], 0.0, None) ** 2
            elif _mode == "log":
                _Y_pool_raw[:, _i] = 10.0 ** _Y_pool_scaled[:, _i]
            else:
                _Y_pool_raw[:, _i] = _Y_pool_scaled[:, _i]

        # Deviation score: same formula as the active-learning loop
        _pool_mean = gen["mean_ref"]
        _pool_std  = gen["std_ref"]
        if _Y_pool_raw.shape[1] == len(_pool_mean):
            _pool_dev = _np3.sum(_np3.abs((_Y_pool_raw - _pool_mean) / _pool_std), axis=1)
        else:
            # mean_ref/std_ref may have fewer columns if only spec_names survive;
            # fall back to simple L2 in normalized pred space
            _pool_dev = _np3.sqrt(_np3.sum(_Y_pool_norm ** 2, axis=1))

        _pred_failures = int(_np3.sum(_pool_dev > TARGET_SIGMA))
        print(f"  Pool predicted failures (dev > {TARGET_SIGMA}σ): {_pred_failures:,} "
              f"/ {len(_X_pool_raw):,} "
              f"({100.0 * _pred_failures / max(len(_X_pool_raw), 1):.4f}%)", flush=True)

        _worst_k = min(_K_WORST, len(_X_pool_raw))
        _worst_idx = _np3.argpartition(_pool_dev, -_worst_k)[-_worst_k:]
        _worst_idx = _worst_idx[_np3.argsort(_pool_dev[_worst_idx])[::-1]]

        _pool_cols = list(exp.active_params) if 'exp' in locals() else [f"p{i}" for i in range(_X_pool_raw.shape[1])]
        _worst_df = pd.DataFrame(_X_pool_raw[_worst_idx], columns=_pool_cols)
        _worst_df.insert(0, "pool_position",    _worst_idx)
        _worst_df.insert(1, "verify_mc_seed",   _worst_idx + 1)
        _worst_df.insert(2, "pred_deviation",   _pool_dev[_worst_idx])
        _worst_df.insert(3, "pred_failure",     _pool_dev[_worst_idx] > TARGET_SIGMA)
        for _i, _sn in enumerate(spec_names):
            _worst_df[f"pred_{_sn}"] = _Y_pool_raw[_worst_idx, _i]

        _worst_df.to_csv(_pool_scores_csv, index=False)
        print(f"  Top-{_worst_k} worst pool candidates saved → {_pool_scores_csv}", flush=True)
        print(f"  (Pool position = 0-indexed row; verify_mc_seed = pool_position + 1,",
              flush=True)
        print(f"   matching the single-call verify-MC run with monte_seed=1, monte={len(_X_pool_raw):,}.)",
              flush=True)

        print("\n  Top-10 worst predicted pool candidates:", flush=True)
        print(f"  {'Pool pos':>10}  {'MC seed':>8}  {'Pred dev':>10}  {'Failure?':>8}",
              flush=True)
        for _, _row in _worst_df.head(10).iterrows():
            print(f"  {int(_row['pool_position']):>10}  "
                  f"{int(_row['verify_mc_seed']):>8}  "
                  f"{_row['pred_deviation']:>10.3f}  "
                  f"{'YES' if _row['pred_failure'] else 'no':>8}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 5 — VIRTUAL SAMPLE GENERATION AND INFERENCE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f"STEP 5: Generating {N_VIRTUAL:,} Virtual Samples", flush=True)
    print("=" * 60, flush=True)

    # Cache Killer: WIPE EVERYTHING to prevent ghost data
    import shutil
    pkl_res_path = os.path.join(OUTDIR, "fastmc_results.pkl")
    if not getattr(args, "view_results", None):
        print(f"  Purging stale caches in {OUTDIR}...", flush=True)
        for f in ["fastmc_results.pkl", f"X_virtual_{N_VIRTUAL}.csv"]:
            p = os.path.join(OUTDIR, f)
            if os.path.exists(p): os.remove(p)

    virtual_csv = os.path.join(OUTDIR, f"X_virtual_{N_VIRTUAL}.csv")
    X_virtual_clean = None

    if os.path.exists(virtual_csv):
        print(f"  Loading cached virtual samples from {virtual_csv}...", flush=True)
        tmp = pd.read_csv(virtual_csv).values.astype("float32")
        if tmp.shape[1] == X_clean.shape[1]:
            X_virtual_clean = tmp
        else:
            print(f"  Shape mismatch ({tmp.shape[1]} vs {X_clean.shape[1]}). Regenerating...", flush=True)

    if X_virtual_clean is None:
        # Generation and inference in chunks to handle massive targets (5-sigma+) without OOM
        print(f"  Generating and predicting {N_VIRTUAL:,} virtual points...", flush=True)
        CHUNK_SIZE = 500000
        Y_virtual_list = []
        
        # Prepare statistical parameters for correlated generation
        GEN_STRETCH = max(WIDENING, TARGET_SIGMA / 3.0)
        if pca is not None:
            pca_stds = np.sqrt(pca.explained_variance_)
        else:
            # If PCA bypassed (raw bypass or supervised feature selection), draw
            # correlated candidates from the covariance of the FULL cleaned/scaled
            # feature set (not just the selected top-K) so tail sampling stays
            # physically realistic across all parameters; the model input is then
            # reduced via _apply_reduction() right before prediction.
            use_native_gen = False
            if use_native_gen:
                X_norm_full = x_scaler.transform(X_clean)
                cov_matrix = np.cov(X_norm_full, rowvar=False)

        # Auditor Setup: Track top 5 worst-case champions
        n_audit = 5
        top_indices = [] # list of (val, X_params)
        
        for start_i in range(0, N_VIRTUAL, CHUNK_SIZE):
            end_i = min(start_i + CHUNK_SIZE, N_VIRTUAL)
            curr_batch = end_i - start_i
            
            if pca is not None:
                # 1. Generate independent virtual variables in PCA space
                X_c_pca = np.random.normal(
                    loc=0.0, scale=pca_stds * GEN_STRETCH,
                    size=(curr_batch, pca.n_components_)
                ).astype("float32")
                # 2. Decode back to correlated, physical space
                X_c_norm = pca.inverse_transform(X_c_pca)
                X_chunk  = x_scaler.inverse_transform(X_c_norm)
            elif use_native_gen:
                # Generate directly in native raw-feature space using the declared
                # distribution's correlation, then forward-transform via x_scaler --
                # the same path training data went through -- to get the model input.
                X_chunk = np.random.multivariate_normal(
                    mean=np.zeros(X_clean.shape[1]),
                    cov=cov_matrix_native * (GEN_STRETCH**2),
                    size=curr_batch
                ).astype("float32")
                X_c_norm = x_scaler.transform(X_chunk)
                X_c_pca = _apply_reduction(X_c_norm, pca, feat_select_idx)
            else:
                # 1. Generate correlated variables using Multivariate Normal
                X_c_norm = np.random.multivariate_normal(
                    mean=np.zeros(X_norm_full.shape[1]),
                    cov=cov_matrix * (GEN_STRETCH**2),
                    size=curr_batch
                ).astype("float32")
                X_c_pca = _apply_reduction(X_c_norm, pca, feat_select_idx)
                X_chunk = x_scaler.inverse_transform(X_c_norm)

            Y_chunk_n   = model.predict(X_c_pca, verbose=0)
            Y_chunk_inv = y_scaler.inverse_transform(Y_chunk_n)

            Y_p = np.empty_like(Y_chunk_inv, dtype="float64")
            for i, mode in enumerate(transform_modes):
                if mode == "sqrt":
                    Y_p[:, i] = Y_chunk_inv[:, i]**2
                elif mode == "log":
                    Y_p[:, i] = 10**Y_chunk_inv[:, i]
                else:
                    Y_p[:, i] = np.maximum(Y_chunk_inv[:, i], 1e-15)
            Y_p = np.ascontiguousarray(Y_p, dtype="float64")

            Y_virtual_list.append(Y_p)
            
            # Local champion search (for spec index 0)
            loc_idx = np.argmax(Y_p[:, 0])
            loc_val = Y_p[loc_idx, 0]
            top_indices.append((loc_val, X_chunk[loc_idx]))
            top_indices.sort(key=lambda x: x[0], reverse=True)
            top_indices = top_indices[:n_audit]
            
        Y_virtual = np.ascontiguousarray(np.vstack(Y_virtual_list))
    else:
        # If loaded from cache, we still need to run inference
        print(f"  Predicting {N_VIRTUAL:,} virtual points from cached X...", flush=True)
        X_c_norm = x_scaler.transform(X_virtual_clean)
        X_c_pca = _apply_reduction(X_c_norm, pca, feat_select_idx)
        Y_virtual_inv = y_scaler.inverse_transform(model.predict(X_c_pca, verbose=0))

        Y_virtual = np.empty_like(Y_virtual_inv, dtype="float64")
        for i, mode in enumerate(transform_modes):
            if mode == "sqrt":
                Y_virtual[:, i] = Y_virtual_inv[:, i]**2
            elif mode == "log":
                Y_virtual[:, i] = 10**Y_virtual_inv[:, i]
            else:
                Y_virtual[:, i] = np.maximum(Y_virtual_inv[:, i], 1e-15)

    v_mean = np.mean(Y_virtual, axis=0)
    v_std  = np.std(Y_virtual,  axis=0)

    # -----------------------------------------------------------------------
    # STEP 6 — GROUND TRUTH AUDIT (REAL SPICE SEEDS)
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # STEP 6 — GROUND TRUTH AUDIT (REAL SPICE SEEDS)
    # -----------------------------------------------------------------------
    if not getattr(args, "view_results", None):
        print("\n" + "=" * 60)
        if args.audit_pool > 0:
            print(f"STEP 6: Industry Standard Audit (SPICE Pool size = {args.audit_pool:,})")
        else:
            print("STEP 6: Ground Truth Audit (Real SPICE Seeds)")
        print("=" * 60)
        
        audit_folder = os.path.join(OUTDIR, f"extreme_failures_sigma_{TARGET_SIGMA}")
        os.makedirs(audit_folder, exist_ok=True)
        n_audit = 5
        
        if args.audit_pool > 0:
            print(f"  Generating a fresh SPICE .op pool of {args.audit_pool:,} real seeds...")
            from experiment import Experiment
            exp_audit = Experiment(SPICE_TB, workdir=WORKDIR, debug=False, simulator=SIMULATOR, ngspice_cmd=NGSPICE_BIN)

            # Force the auditor to use exactly the same features we trained on
            if 'X_train_df' in locals():
                exp_audit.active_params = list(X_train_df.columns)
            
            X_audit_pool_df = exp_audit.execute_generation(n_samples=args.audit_pool)
            print("  Predicting delays for the SPICE pool...")
            
            # Apply exactly the same preprocessing pipeline as training
            X_audit_raw = X_audit_pool_df.values.astype("float64")
            X_audit_clean = selector.transform(X_audit_raw)
            X_audit_norm = x_scaler.transform(X_audit_clean)
            X_audit_pca = _apply_reduction(X_audit_norm, pca, feat_select_idx)

            Y_audit_pred_norm = model.predict(X_audit_pca, verbose=0)
            Y_audit_pred_inv  = y_scaler.inverse_transform(Y_audit_pred_norm)

            Y_audit_pred = np.empty_like(Y_audit_pred_inv)
            for i, mode in enumerate(transform_modes):
                if mode == "sqrt":
                    Y_audit_pred[:, i] = Y_audit_pred_inv[:, i]**2
                elif mode == "log":
                    Y_audit_pred[:, i] = 10**Y_audit_pred_inv[:, i]
                else:
                    Y_audit_pred[:, i] = Y_audit_pred_inv[:, i]
                
            top_mlp_indices = np.argsort(Y_audit_pred[:, 0])[::-1][:n_audit]
            
            print(f"\nAUDIT REPORT ({spec_names[0]} Absolute Tails from {args.audit_pool:,} Real SPICE Seeds):")
            print(f"{'Rank':<5} | {'MLP Prediction':<15} | {'SPICE Truth':<15} | {'Error %':<10} | {'Seed #':<10}")
            print("-" * 75)
            
            from Simulator import Simulator
            for rank, idx in enumerate(top_mlp_indices):
                worst_seed = 10007 + idx  # The exact seed offset used in Simulator.py
                pred_val   = Y_audit_pred[idx, 0]
                
                audit_tb = os.path.join(audit_folder, f"worst_case_rank_{rank+1}_seed_{worst_seed}.sp")
                with open(SPICE_TB, "r") as f:
                    content = f.read()
                    
                import re
                _END_RE = re.compile(r"^\.end\b", re.IGNORECASE | re.MULTILINE)
                audit_control = f"\n.control\n  set seed = {worst_seed}\n  run\n  quit\n.endc\n"
                if _END_RE.search(content):
                    content = _END_RE.sub(f"{audit_control}\n.end", content)
                else:
                    content += f"\n{audit_control}\n.end\n"
                
                with open(audit_tb, "w") as f:
                    f.write(content)
                    
                sim_audit = Simulator(audit_tb, WORKDIR)
                logfile = sim_audit.run(seed=worst_seed)
                
                actual_val = float('nan')
                if os.path.exists(logfile):
                    audit_log_save = os.path.join(audit_folder, f"worst_case_rank_{rank+1}_seed_{worst_seed}.log")
                    import shutil
                    shutil.copy2(logfile, audit_log_save)
                    
                    with open(logfile, "r") as f:
                        from parser import NgspiceParser
                        p = NgspiceParser()
                        _, targs = p.parse_log(f.read())
                        if spec_names[0] in targs:
                            actual_val = targs[spec_names[0]]
                            
                if not np.isnan(actual_val):
                    err = abs(pred_val - actual_val) / (abs(actual_val) + 1e-20) * 100
                    print(f"#{rank+1:<4} | {pred_val:<15.4e} | {actual_val:<15.4e} | {err:<10.2f}% | {worst_seed:<10}")
                else:
                    print(f"#{rank+1:<4} | {pred_val:<15.4e} | ERROR SIMULATING  | ---        | {worst_seed:<10}")
                    
            print("-" * 75)
            print(f"  Audit Complete. Verification netlists saved to: {audit_folder}\n")

        else:
            if seeds_array is None or 'X_train_df' not in locals():
                print("  [Audit] Training-set data unavailable (e.g. loaded via --load-model). "
                      "Cannot run SPICE verification.")
            else:
                print(f"  Finding the Top {n_audit} most extreme REAL SPICE seeds from the {len(Y_raw)} training points...")

                top_mlp_indices = np.argsort(Y_train_pred[:, 0])[::-1][:n_audit]

                from experiment import Experiment
                exp_audit = Experiment(SPICE_TB, workdir=WORKDIR, debug=False, simulator=SIMULATOR,
                                        ngspice_cmd=NGSPICE_BIN)
                exp_audit.active_params = list(X_train_df.columns)

                X_top_df = pd.DataFrame(X_raw[top_mlp_indices], columns=list(X_train_df.columns))
                X_new_df, Y_new_df = exp_audit.execute_audit(X_top_df, audit_dir=audit_folder)

                # Extrapolation context (informational only, purely data-driven -- does
                # NOT rely on interpreting the netlist's declared distribution
                # parameters): how far each audited point's most extreme parameter sits
                # relative to the largest magnitude the native (unstretched)
                # sampling ever produced for that same parameter.
                native_max_ref = None
                if 'exp' in locals() and hasattr(exp, 'param_distributions') and exp.param_distributions and (seeds_array < 100000).any():
                    native_mask = seeds_array < 100000
                    col_names_full = list(X_train_df.columns)
                    native_abs_max = np.abs(X_raw[native_mask]).max(axis=0)
                    native_max_ref = dict(zip(col_names_full, native_abs_max))

                print(f"\nAUDIT REPORT ({spec_names[0]} Absolute Tails from Training Seeds -- freshly re-simulated):")
                print(f"{'Rank':<5} | {'MLP Prediction':<15} | {'SPICE Truth':<15} | {'Error %':<10} | {'Row ID':<10}")
                print("-" * 75)

                n_verified = min(len(Y_new_df), n_audit)
                for rank in range(n_verified):
                    idx = top_mlp_indices[rank]
                    row_id = int(seeds_array[idx])
                    pred_val = Y_train_pred[idx, 0]
                    actual_val = Y_new_df[spec_names[0]].iloc[rank] if spec_names[0] in Y_new_df.columns else float("nan")

                    if np.isfinite(actual_val):
                        err = abs(pred_val - actual_val) / (abs(actual_val) + 1e-20) * 100
                        print(f"#{rank+1:<4} | {pred_val:<15.4e} | {actual_val:<15.4e} | {err:<10.2f}% | {row_id:<10}")
                    else:
                        print(f"#{rank+1:<4} | {pred_val:<15.4e} | ERROR SIMULATING  | ---        | {row_id:<10}")

                    if native_max_ref:
                        row_vals = X_raw[idx]
                        ratios = [
                            (abs(row_vals[j]) / native_max_ref[name], name, row_vals[j], native_max_ref[name])
                            for j, name in enumerate(col_names_full)
                            if native_max_ref[name] > 1e-12
                        ]
                        if ratios:
                            worst_ratio, worst_name, worst_val, worst_native = max(ratios, key=lambda r: r[0])
                            print(f"       extrapolation: {worst_name[:50]} = {worst_val:.3f} "
                                  f"({worst_ratio:.2f}x the largest value from native "
                                  f"sampling for it, {worst_native:.3f})")

                if n_verified < n_audit:
                    print(f"  [Warning] Only {n_verified}/{n_audit} points returned a verified result.")

                print("-" * 75)
                print(f"  Audit Complete. Model accuracy re-verified via fresh simulation of the "
                      f"top {n_verified} extreme points.")
                print(f"  Verification artifacts saved under: {audit_folder}\n")

    # -----------------------------------------------------------------------
    # STEP 7 — BRUTE-FORCE MC VERIFICATION (optional, --verify-mc)
    # -----------------------------------------------------------------------
    if getattr(args, "verify_mc", False):
        print("\n" + "=" * 60, flush=True)
        print("STEP 7: Brute-Force MC Verification", flush=True)
        print("=" * 60, flush=True)

        N_VERIFY = getattr(args, "verify_seeds", None) or N_NATIVE_POOL_AUTO
        K_VERIFY = getattr(args, "verify_k", 50)

        # Resolve experiment object — available in fresh-run, needs -spice in load-model mode
        try:
            _exp_v = exp
        except NameError:
            _exp_v = None
        if _exp_v is None:
            if not getattr(args, "spice", None):
                print("  [SKIP] --verify-mc requires -spice TESTBENCH.\n"
                      "  Re-run with: -spice path/to/tb.sp --load-model DIR --verify-mc",
                      flush=True)
            else:
                print(f"  Creating experiment from {args.spice}...", flush=True)
                from experiment import Experiment as _Experiment
                _exp_v = _Experiment(
                    args.spice,
                    workdir=WORKDIR,
                    debug=False,
                    simulator=SIMULATOR,
                    ngspice_cmd=NGSPICE_BIN,
                    ngspice_np=NGSPICE_NP,
                )

        if _exp_v is not None:
            print("  [INFO] Brute-force MC verification via ngspice_mc: running "
                  f"{N_VERIFY:,} seeds sequentially.", flush=True)
            print(f"  Generating parameter matrix ({N_VERIFY:,} seeds)...", flush=True)
            X_mc_df, Y_mc_df = _exp_v.execute_generation(
                n_samples=N_VERIFY, monte_seed=1
            )
            print(f"  X (parameters): {X_mc_df.shape}, Y (measurements): {Y_mc_df.shape}",
                  flush=True)

            # Column alignment: use stored training columns (fresh run or from meta.pkl)
            try:
                _x_cols = list(X_train_df.columns)
            except NameError:
                _x_cols = meta.get("X_train_columns") or _exp_v.active_params

            _spec_cols_v = [c for c in spec_names if c in Y_mc_df.columns]
            if not _spec_cols_v:
                print(f"  [SKIP] None of spec_names {spec_names} found in MC output columns. "
                      f"Available: {list(Y_mc_df.columns)}", flush=True)
            else:
                X_mc_raw = X_mc_df.reindex(columns=_x_cols, fill_value=0.0).values.astype("float64")
                Y_mc_raw = Y_mc_df[_spec_cols_v].values.astype("float64")

                # Valid rows: non-zero X and non-nan, positive Y
                _mc_valid = (
                    (X_mc_raw != 0).any(axis=1)
                    & ~_np.isnan(Y_mc_raw).any(axis=1)
                    & (Y_mc_raw > 0).all(axis=1)
                )
                print(f"  Valid MC seeds: {_mc_valid.sum():,} / {len(_mc_valid):,}", flush=True)

                if _mc_valid.sum() < K_VERIFY:
                    print(f"  [SKIP] Valid seeds ({_mc_valid.sum()}) < K_VERIFY ({K_VERIFY}). "
                          f"Increase --verify-seeds.", flush=True)
                else:
                    X_mc_v  = X_mc_raw[_mc_valid]
                    Y_mc_act = Y_mc_raw[_mc_valid]

                    # Apply the same surrogate preprocessing pipeline
                    X_mc_cl  = selector.transform(X_mc_v)
                    X_mc_nr  = x_scaler.transform(X_mc_cl)
                    if feat_select_idx is not None:
                        X_mc_in = X_mc_nr[:, feat_select_idx]
                    elif pca is not None:
                        X_mc_in = pca.transform(X_mc_nr)
                    else:
                        X_mc_in = X_mc_nr

                    print(f"  Scoring {_mc_valid.sum():,} seeds with surrogate...", flush=True)
                    _Y_pred_t = y_scaler.inverse_transform(
                        model.predict(X_mc_in, batch_size=4096, verbose=0)
                    )
                    # Undo transform_modes → physical spec units
                    Y_mc_pred = _np.zeros_like(_Y_pred_t)
                    for _i, _mode in enumerate(transform_modes):
                        if _i >= _Y_pred_t.shape[1]:
                            break
                        if _mode == "log":
                            Y_mc_pred[:, _i] = 10.0 ** _Y_pred_t[:, _i]
                        elif _mode == "sqrt":
                            Y_mc_pred[:, _i] = _Y_pred_t[:, _i] ** 2
                        else:
                            Y_mc_pred[:, _i] = _Y_pred_t[:, _i]

                    # Score both by sum-of-absolute-normalized-deviations
                    # Reference is the brute-force MC distribution (ground truth)
                    _mc_mean = Y_mc_act.mean(axis=0)
                    _mc_std  = Y_mc_act.std(axis=0)
                    _mc_std[_mc_std < 1e-30] = 1e-30

                    actual_scores = _np.sum(_np.abs((Y_mc_act  - _mc_mean) / _mc_std), axis=1)
                    pred_scores   = _np.sum(_np.abs((Y_mc_pred - _mc_mean) / _mc_std), axis=1)

                    actual_top_k = set(_np.argsort(actual_scores)[-K_VERIFY:].tolist())
                    pred_top_k   = set(_np.argsort(pred_scores)[-K_VERIFY:].tolist())
                    overlap = len(actual_top_k & pred_top_k)

                    from scipy.stats import spearmanr as _spearmanr
                    _rho, _pval = _spearmanr(actual_scores, pred_scores)

                    # Confirmed failure rate from brute-force MC (using sum-of-devs > TARGET_SIGMA)
                    _n_fail_mc = int(_np.sum(actual_scores > TARGET_SIGMA))
                    _fail_rate = _n_fail_mc / _mc_valid.sum()
                    _expected_pf = 2.0 * _norm.sf(TARGET_SIGMA)

                    print(f"\n  Brute-Force MC Verification Results", flush=True)
                    print(f"  " + "-" * 55, flush=True)
                    print(f"  {'Seeds simulated':<35}: {N_VERIFY:,}", flush=True)
                    print(f"  {'Valid seeds':<35}: {_mc_valid.sum():,}", flush=True)
                    print(f"  {'Confirmed failures (dev>{TARGET_SIGMA:.1f}σ)':<35}: "
                          f"{_n_fail_mc:,}  ({_fail_rate*100:.4f}%  "
                          f"vs theoretical {_expected_pf*100:.4f}%)", flush=True)
                    print(f"  {'Precision@'+str(K_VERIFY):<35}: "
                          f"{overlap}/{K_VERIFY}  ({overlap/K_VERIFY*100:.1f}%)", flush=True)
                    print(f"  {'Spearman rank corr (ρ)':<35}: {_rho:.4f}  (p={_pval:.2e})",
                          flush=True)

                    # Top-K table
                    _actual_sorted = _np.argsort(actual_scores)[::-1][:K_VERIFY]
                    print(f"\n  Top-{K_VERIFY} seeds ranked by actual deviation:", flush=True)
                    print(f"  {'Rank':>5}  {'Actual dev':>11}  {'Pred dev':>11}  "
                          f"{'In pred top-'+str(K_VERIFY):>18}", flush=True)
                    print(f"  " + "-" * 55, flush=True)
                    for _r, _idx in enumerate(_actual_sorted[:20]):
                        _in_pred = "YES" if _idx in pred_top_k else ""
                        print(f"  {_r+1:>5}  {actual_scores[_idx]:>11.3f}  "
                              f"{pred_scores[_idx]:>11.3f}  {_in_pred:>18}", flush=True)
                    if K_VERIFY > 20:
                        print(f"  ... (showing top-20 of {K_VERIFY})", flush=True)

                    # ---------------------------------------------------------------
                    # Final surrogate failure detection vs. verify-mc ground truth
                    # pred_scores / actual_scores are already computed on the same
                    # N_VERIFY seeds → threshold at TARGET_SIGMA for a full confusion matrix.
                    # ---------------------------------------------------------------
                    _pred_fail  = pred_scores   > TARGET_SIGMA
                    _actual_fail = actual_scores > TARGET_SIGMA
                    _ftp = int(_np.sum( _pred_fail &  _actual_fail))
                    _ffp = int(_np.sum( _pred_fail & ~_actual_fail))
                    _ffn = int(_np.sum(~_pred_fail &  _actual_fail))
                    _ftn = int(_np.sum(~_pred_fail & ~_actual_fail))
                    _fprec = _ftp / (_ftp + _ffp) if _ftp + _ffp > 0 else 0.0
                    _frec  = _ftp / (_ftp + _ffn) if _ftp + _ffn > 0 else 0.0
                    _ff1   = (2 * _fprec * _frec / (_fprec + _frec)
                              if _fprec + _frec > 0 else 0.0)

                    print(f"\n  [Final Surrogate] Failure detection vs. brute-force MC "
                          f"(threshold: dev > {TARGET_SIGMA}σ across {_mc_valid.sum():,} seeds)",
                          flush=True)
                    print(f"  " + "-" * 55, flush=True)
                    print(f"  {'True  Positives  (surrogate+, MC+)':<38}: {_ftp:>6}", flush=True)
                    print(f"  {'False Positives  (surrogate+, MC-)':<38}: {_ffp:>6}", flush=True)
                    print(f"  {'False Negatives  (surrogate-, MC+)':<38}: {_ffn:>6}", flush=True)
                    print(f"  {'True  Negatives  (surrogate-, MC-)':<38}: {_ftn:>6}", flush=True)
                    print(f"  {'Precision':<38}: {_fprec*100:>5.1f}%  "
                          f"(of FastMC predicted failures that are real)", flush=True)
                    print(f"  {'Recall':<38}: {_frec*100:>5.1f}%  "
                          f"(of real failures that FastMC caught)", flush=True)
                    print(f"  {'F1 score':<38}: {_ff1:>5.3f}", flush=True)

                    # Save full comparison CSV (add failure columns)
                    _verify_csv = os.path.join(OUTDIR, f"mc_verify_{N_VERIFY}_seeds.csv")
                    _df_v = pd.DataFrame({
                        "seed":            _np.arange(1, len(actual_scores) + 1),
                        "actual_deviation": actual_scores,
                        "pred_deviation":   pred_scores,
                        "actual_failure":   _actual_fail,
                        "pred_failure":     _pred_fail,
                        "in_actual_top_k":  [i in actual_top_k for i in range(len(actual_scores))],
                        "in_pred_top_k":    [i in pred_top_k   for i in range(len(actual_scores))],
                    })
                    for _si, _sn in enumerate(_spec_cols_v):
                        if _si < Y_mc_act.shape[1]:
                            _df_v[f"actual_{_sn}"] = Y_mc_act[:, _si]
                        if _si < Y_mc_pred.shape[1]:
                            _df_v[f"pred_{_sn}"]   = Y_mc_pred[:, _si]
                    _df_v.to_csv(_verify_csv, index=False)
                    print(f"\n  Verification data saved → {_verify_csv}", flush=True)

                    # ---------------------------------------------------------------
                    # Convergence plot: FastMC vs Traditional MC
                    # X = cumulative seeds simulated
                    # Y = cumulative failures detected (dev > TARGET_SIGMA)
                    # ---------------------------------------------------------------
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as _plt

                        # Traditional MC curve — cumulative failures in natural seed order
                        _tmc_fail_arr = (actual_scores > TARGET_SIGMA).astype(int)
                        _tmc_cum      = _np.cumsum(_tmc_fail_arr)
                        _tmc_x        = _np.arange(1, len(_tmc_cum) + 1)

                        # FastMC curve — cumulative failures in simulation order
                        _fmc_x = [0];  _fmc_y = [0]
                        # Initial batch failures
                        _spec_fmc = [c for c in spec_names if c in Y_train_df.columns]
                        if ('gen' in locals() and gen is not None
                                and len(_spec_fmc) == len(gen["mean_ref"])):
                            _Y_init = (Y_train_df[_spec_fmc].iloc[:N_INITIAL]
                                       .values.astype("float64"))
                            _init_devs = _np.sum(
                                _np.abs((_Y_init - gen["mean_ref"]) / gen["std_ref"]), axis=1)
                            _fmc_y_start = int(_np.sum(_init_devs > TARGET_SIGMA))
                        else:
                            _fmc_y_start = 0
                        _fmc_x.append(N_INITIAL);  _fmc_y.append(_fmc_y_start)

                        # AL phase — one step per audit point in the order they were simulated
                        _cum_fmc = _fmc_y_start
                        for _rec in fastmc_audit_records:
                            _cum_fmc += int(_rec.get("fastmc_failure", False))
                            _fmc_x.append(_fmc_x[-1] + 1)
                            _fmc_y.append(_cum_fmc)

                        fig, ax = _plt.subplots(figsize=(10, 6))

                        # Downsample traditional MC for large N to keep plot file small
                        _step = max(1, len(_tmc_x) // 2000)
                        ax.plot(_tmc_x[::_step], _tmc_cum[::_step],
                                color="steelblue", linewidth=1.5,
                                label=f"Traditional MC  ({N_VERIFY:,} seeds, "
                                      f"{_n_fail_mc:,} failures)")

                        ax.step(_fmc_x, _fmc_y, color="tomato", linewidth=2.0, where="post",
                                label=f"FastMC  ({len(_fmc_x)-1 + N_INITIAL} seeds, "
                                      f"{_fmc_y[-1]} failures found)")

                        ax.axvline(N_TRAIN_SEEDS, color="tomato", linestyle="--", alpha=0.4,
                                   label=f"FastMC budget ({N_TRAIN_SEEDS:,} sims)")

                        ax.set_xlabel("Total Seeds Simulated", fontsize=13)
                        ax.set_ylabel(f"Cumulative Failures  (deviation > {TARGET_SIGMA}σ)",
                                      fontsize=13)
                        ax.set_title(
                            f"FastMC vs Traditional MC — Failure Discovery Curve\n"
                            f"Target: {TARGET_SIGMA}σ  |  "
                            f"Valid MC seeds: {_mc_valid.sum():,}  |  "
                            f"MC failure rate: {_fail_rate*100:.4f}%",
                            fontsize=12,
                        )
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(left=0)
                        ax.set_ylim(bottom=0)

                        _plot_path = os.path.join(OUTDIR, "fastmc_vs_mc_convergence.png")
                        fig.savefig(_plot_path, dpi=150, bbox_inches="tight")
                        _plt.close(fig)
                        print(f"  Convergence plot saved → {_plot_path}", flush=True)
                    except Exception as _plot_err:
                        print(f"  [Warning] Could not save convergence plot: {_plot_err}",
                              flush=True)

                    # ---------------------------------------------------------------
                    # Active-learning efficiency: audit points picked during training
                    # vs. their corresponding verify-mc seeds (answers "did AL hunt
                    # real failures efficiently?" — separate from final surrogate accuracy)
                    # Each pool position P (0-indexed) = verify-mc row P = seed P+1
                    # (both were generated with monte_seed=1 in the same order).
                    # ---------------------------------------------------------------
                    if fastmc_audit_records:
                        print(f"\n  [AL Phase Efficiency] {len(fastmc_audit_records)} points forced "
                              f"during active learning vs. their exact verify-mc seeds "
                              f"(pool pos P = seed P+1, verified bit-for-bit aligned)...",
                              flush=True)

                        # Deviation for ALL Y_mc_df rows (row P → seed P+1) using the
                        # same reference already computed from the valid MC subset.
                        _spec_cols_m = [c for c in spec_names if c in Y_mc_df.columns]
                        _Y_mc_all = Y_mc_df[_spec_cols_m].values.astype("float64") if _spec_cols_m else None
                        if _Y_mc_all is not None:
                            _mc_all_devs = _np.sum(
                                _np.abs((_Y_mc_all - _mc_mean) / _mc_std), axis=1
                            )
                            _mc_all_fail = _mc_all_devs > TARGET_SIGMA

                            print(f"\n  {'#':>5}  {'PoolPos':>8}  {'MC seed':>8}  "
                                  f"{'FastMC pred':>12}  {'FastMC dev':>11}  {'FM fail?':>9}  "
                                  f"{'MC dev':>10}  {'MC fail?':>9}  {'Match?':>7}", flush=True)
                            print(f"  " + "-" * 94, flush=True)

                            tp = fp = tn = fn = 0
                            _match_rows = []
                            for _ai, _rec in enumerate(fastmc_audit_records):
                                _pos  = _rec["pool_position"]
                                _seed = _pos + 1
                                _fm_fail = _rec.get("fastmc_failure", False)
                                _fm_dev  = _rec.get("fastmc_deviation", float("nan"))
                                _fm_pred = _rec.get("predicted_score", float("nan"))
                                if _pos < len(_mc_all_devs):
                                    _mc_dev  = float(_mc_all_devs[_pos])
                                    _mc_fail = bool(_mc_all_fail[_pos])
                                else:
                                    _mc_dev, _mc_fail = float("nan"), False

                                _match = (_fm_fail == _mc_fail)
                                if _fm_fail and _mc_fail:        tp += 1
                                elif _fm_fail and not _mc_fail:  fp += 1
                                elif not _fm_fail and _mc_fail:  fn += 1
                                else:                             tn += 1

                                print(f"  {_ai+1:>5}  {_pos:>8}  {_seed:>8}  "
                                      f"{_fm_pred:>12.3f}  {_fm_dev:>11.3f}  "
                                      f"{'YES' if _fm_fail else '':>9}  "
                                      f"{_mc_dev:>10.3f}  {'YES' if _mc_fail else '':>9}  "
                                      f"{'OK' if _match else 'MISMATCH':>7}", flush=True)

                                _match_rows.append({
                                    "pool_position":    _pos,
                                    "verify_mc_seed":   _seed,
                                    "fastmc_pred_score": _fm_pred,
                                    "fastmc_deviation": _fm_dev,
                                    "fastmc_failure":   _fm_fail,
                                    "verifymc_deviation": _mc_dev,
                                    "verifymc_failure": _mc_fail,
                                    "match":            _match,
                                })

                            _n_rec = len(fastmc_audit_records)
                            print(f"\n  Confusion matrix (FastMC predictor / verify-mc ground truth):", flush=True)
                            print(f"    True  Positives  (FastMC+, MC+): {tp:>4}", flush=True)
                            print(f"    False Positives  (FastMC+, MC-): {fp:>4}", flush=True)
                            print(f"    False Negatives  (FastMC-, MC+): {fn:>4}", flush=True)
                            print(f"    True  Negatives  (FastMC-, MC-): {tn:>4}", flush=True)
                            if tp + fp > 0:
                                print(f"    Precision (of FastMC failures that are real): "
                                      f"{tp/(tp+fp)*100:.1f}%", flush=True)
                            if tp + fn > 0:
                                print(f"    Recall    (of real failures FastMC caught):   "
                                      f"{tp/(tp+fn)*100:.1f}%", flush=True)

                            _match_csv = os.path.join(OUTDIR, "fastmc_failure_matching.csv")
                            pd.DataFrame(_match_rows).to_csv(_match_csv, index=False)
                            print(f"  Failure matching saved → {_match_csv}", flush=True)

    # Filter specs for display if requested
    if SHOW_SPECS:
        display_indices = [i for i, name in enumerate(spec_names) if name in SHOW_SPECS]
        if not display_indices:
            print(f"\n[FastMC] WARNING: None of the requested --show-specs {SHOW_SPECS} found in model. Showing all instead.", flush=True)
            display_indices = list(range(len(spec_names)))
    else:
        display_indices = list(range(len(spec_names)))

    s_low_title  = f"{args.sigma}s-low"
    s_high_title = f"{args.sigma}s-high"
    print(f"\n  {'Spec':<25} {'Mean':>14} {'Std':>14} {s_low_title:>14} {s_high_title:>14}")
    print("  " + "-" * 79)
    for i in display_indices:
        name = spec_names[i]
        low  = v_mean[i] - args.sigma * v_std[i]
        high = v_mean[i] + args.sigma * v_std[i]
        print(f"  {name:<25} {v_mean[i]:>14.4e} {v_std[i]:>14.4e} {low:>14.4e} {high:>14.4e}")

    # -----------------------------------------------------------------------
    # STEP 6 — PLOTS AND EXPORT
    # -----------------------------------------------------------------------
    if EXPORT_CSV:
        if len(Y_virtual) > 1000000:
            print(f"\n  [FastMC] WARNING: Sample count too large for CSV ({len(Y_virtual)} rows).")
            print(f"  Exporting first 1M samples only to {EXPORT_CSV}")
            pd.DataFrame(Y_virtual[:1000000], columns=spec_names).to_csv(EXPORT_CSV, index=False)
        else:
            df_v = pd.DataFrame(Y_virtual, columns=spec_names)
            df_v.to_csv(EXPORT_CSV, index=False)
        print(f"  Done.")

    plot_results(
        spec_names, Y_raw, Y_virtual,
        Y_val_real, Y_val_pred, 
        all_r2, valid_n,
        SHOW_SPECS, SHOW_PLOTS, OUTDIR, 
        history=history, best_epoch=best_epoch,
        sigma=args.sigma
    )

    print("\n[FastMC] Done.", flush=True)


def plot_results(spec_names, Y_raw, Y_virtual, Y_val_real, Y_val_pred, 
                all_r2, valid_n, SHOW_SPECS, show_plots, outdir, 
                history=None, best_epoch=None, sigma=3.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import joblib

    v_mean = np.mean(Y_virtual, axis=0)
    v_std  = np.std(Y_virtual,  axis=0)

    # Filter specs for display if requested
    if SHOW_SPECS:
        display_indices = [i for i, name in enumerate(spec_names) if name in SHOW_SPECS]
        if not display_indices:
            print(f"\n[FastMC] WARNING: None of the requested --show-specs {SHOW_SPECS} found. Showing all.", flush=True)
            display_indices = list(range(len(spec_names)))
    else:
        display_indices = list(range(len(spec_names)))

    # (Table removed to prevent mismatch — using Auditor diagnostics below)

    n_disp  = len(display_indices)
    n_cols  = min(3, n_disp)
    n_rows  = (n_disp + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(6 * n_cols, 5 * 2 * n_rows))
    if n_disp == 1:
        axes = axes.reshape(2, 1)

    # Global Styling
    plt.rcParams.update({'font.size': 10, 'axes.labelpad': 8})
    
    for i, idx in enumerate(display_indices):
        name = spec_names[idx]
        row_block = (i // n_cols) * 2
        col = i % n_cols
        
        ax_parity = axes[row_block, col]
        ax_dist   = axes[row_block + 1, col]

        # --- 1. Parity Plot Upgrade ---
        if Y_val_real is not None and Y_val_pred is not None:
            r2 = all_r2[idx] if all_r2 is not None else 0.0
            ax_parity.scatter(Y_val_real[:, idx], Y_val_pred[:, idx], 
                             alpha=0.6, s=25, color="#3498db", edgecolors="k", linewidth=0.5, label=f"R²={r2:.3f}")
            
            # Add ideal 1:1 line
            mn = min(Y_val_real[:, idx].min(), Y_val_pred[:, idx].min())
            mx = max(Y_val_real[:, idx].max(), Y_val_pred[:, idx].max())
            ax_parity.plot([mn, mx], [mn, mx], color="#e74c3c", ls="--", lw=1.5, alpha=0.8, label="Ideal")
            
            ax_parity.set_title(f"{name} | Correlation", fontsize=12, fontweight="bold", pad=10)
            ax_parity.set_xlabel("SPICE Ground Truth", fontsize=10)
            ax_parity.set_ylabel("FastMC Prediction", fontsize=10)
            ax_parity.legend(loc="upper left", fontsize=8, frameon=True)
            ax_parity.grid(True, alpha=0.2)
        else:
            ax_parity.text(0.5, 0.5, "Validation Data\nNot Available", ha="center", va="center", color="gray")

        # --- 2. Distribution Plot Upgrade ---
        spice_vals = Y_raw[:, idx]
        spice_mean, spice_std = np.mean(spice_vals), np.std(spice_vals)
        v_mean_loc = np.mean(Y_virtual[:, idx])
        v_std_loc  = np.std(Y_virtual[:, idx])
        
        v_min, v_max = np.min(Y_virtual[:, idx]), np.max(Y_virtual[:, idx])
        s_min, s_max = np.min(spice_vals), np.max(spice_vals)
        
        x_lo = min(v_min, s_min, spice_mean - sigma * spice_std)
        x_hi = max(v_max, s_max, spice_mean + sigma * spice_std)
        pad = (x_hi - x_lo) * 0.10
        x_lo -= pad
        x_hi += pad

        ax_dist.hist(Y_virtual[:, idx], bins=100, alpha=0.5, range=(x_lo, x_hi),
                color="#2ecc71", density=True, label="FastMC Virtual Pool")
        ax_dist.hist(spice_vals, bins=25, alpha=0.6, range=(x_lo, x_hi),
                color="#7f8c8d", density=True, edgecolor="black", linewidth=0.5, label=f"SPICE Train ({valid_n})")

        # Bounds logic
        target_sigmas = sorted(list(set([3, sigma])))
        sigma_colors = {3: "#e67e22", sigma: "#9b59b6"}
        
        for k in target_sigmas:
            col = sigma_colors.get(k, "#9b59b6")
            ax_dist.axvline(spice_mean - k * spice_std, color=col, lw=1.2, ls=":", alpha=0.7)
            ax_dist.axvline(spice_mean + k * spice_std, color=col, lw=1.2, ls=":", alpha=0.7)
            ax_dist.axvline(v_mean_loc - k * v_std_loc, color=col, lw=2.0, ls="-", label=f"{k}σ Boundary")
            ax_dist.axvline(v_mean_loc + k * v_std_loc, color=col, lw=2.0, ls="-")

        ax_dist.set_xlim(x_lo, x_hi)
        ax_dist.set_title(f"{name} | Tail Resolution", fontsize=12, fontweight="bold", pad=10)
        ax_dist.legend(fontsize=8, loc="upper right", frameon=True)
        ax_dist.grid(True, alpha=0.15)
        ax_dist.set_xlabel("Measured Performance", fontsize=10)

    # Clean up empty slots
    total_slots = n_rows * n_cols
    for j in range(n_disp, total_slots):
        row_block = (j // n_cols) * 2
        col = j % n_cols
        axes[row_block, col].set_axis_off()
        axes[row_block + 1, col].set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.35)

    # Training loss plot (only if data provided)
    fig_loss = None
    if history is not None:
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        ax_loss.plot(history.history["loss"], label="Train")
        ax_loss.plot(history.history["val_loss"], ls="--", label="Val")
        if best_epoch:
            ax_loss.axvline(best_epoch - 1, color="red", lw=1, ls=":")
        ax_loss.set_title("Training Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        fig_loss.tight_layout()

    if show_plots:
        plt.show()
    elif outdir:
        # Save results with unique timestamp to avoid file-lock/stale-cache issues
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(outdir, f"fastmc_results_{ts}.png")
        pkl_path = os.path.join(outdir, f"fastmc_results_{ts}.pkl")
        
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        
        result_bundle = {
            "spec_names": spec_names,
            "Y_raw":      Y_raw,
            "Y_virtual":  Y_virtual,
            "Y_val_real": Y_val_real,
            "Y_val_pred": Y_val_pred,
            "all_r2":     all_r2,
            "valid_n":    valid_n
        }
        joblib.dump(result_bundle, pkl_path)
        
        print(f"\n  Results exported to unique files:")
        print(f"    {png_path}")
        print(f"    {pkl_path}")
        
        if fig_loss:
            fig_loss.savefig(os.path.join(outdir, "fastmc_loss.png"), dpi=150)



# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()

