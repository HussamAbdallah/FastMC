import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures

from Simulator import Simulator, _ANALYSIS_RE
from parser import NgspiceParser


_CONTROL_RE = re.compile(
    r"^\s*\.control\b[\s\S]*?\.endc", re.MULTILINE | re.IGNORECASE
)
_END_RE = re.compile(r"^\s*\.end\b.*", re.MULTILINE | re.IGNORECASE)


def _clean_for_generator(content):
    """Strip control blocks, .end, and all analysis lines."""
    content = _CONTROL_RE.sub("", content)
    content = _END_RE.sub("", content)
    content = _ANALYSIS_RE.sub("", content)
    return content


# ---------------------------------------------------------------------------
# Module-level worker — must be at module level for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _worker(args):
    seed, spfile, workdir, active_params, debug, ngspice_cmd = args
    sim    = Simulator(spfile, workdir, ngspice_cmd=ngspice_cmd)
    parser = NgspiceParser(debug=debug)
    try:
        logfile = sim.run(seed)
        with open(logfile) as f:
            features, targets = parser.parse_log(f.read(),
                                                  relevant_params=active_params)
    except Exception as e:
        print(f"[Worker] Seed {seed} failed: {e}")
        features, targets = {}, {}
    return seed, features, targets


def _mc_worker(args):
    """ngspice_mc parallel worker: full circuit sim for one monte_seed index."""
    seed_k, spfile, workdir, debug, ngspice_cmd = args
    sim    = Simulator(spfile, workdir, ngspice_cmd=ngspice_cmd)
    parser = NgspiceParser(debug=debug)
    try:
        logfile = sim.run_monte_seed(seed_k)
        with open(logfile) as f:
            _, targets = parser.parse_log(f.read(), relevant_params=None)
    except Exception as e:
        print(f"[MC Worker] seed_k={seed_k} failed: {e}")
        targets = {}
    return seed_k, targets


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    def __init__(self, spfile, workdir="./temp", debug=False, simulator="ngspice_mc",
                 ngspice_cmd="ngspice", ngspice_np=4):
        self.spfile          = os.path.abspath(spfile)
        self.workdir         = os.path.abspath(workdir)
        self.debug           = debug
        self.simulator_type  = simulator.lower()
        self.ngspice_cmd     = ngspice_cmd
        self.ngspice_np      = ngspice_np

        self.simulator = Simulator(spfile, workdir, ngspice_cmd=ngspice_cmd)
        self.parser    = NgspiceParser(debug=debug)

        self.active_params     = None
        self.fast_generator_tb = None
        self._mc_params_df     = None  # ngspice_mc: pool DataFrame (0-based index)
        self._mc_executor      = None  # ngspice_mc: persistent pool, created lazily

    def _get_mc_executor(self):
        """Return the persistent ProcessPoolExecutor for ngspice_mc, creating it if needed."""
        if self._mc_executor is None:
            self._mc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.ngspice_np)
        return self._mc_executor

    def __del__(self):
        if self._mc_executor is not None:
            self._mc_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # PEX detection
    # ------------------------------------------------------------------

    def _is_pex_file(self, filepath, rc_threshold=500):
        if not os.path.exists(filepath):
            return False
        rc_count = 0
        with open(filepath, "r") as f:
            for line in f:
                if line.strip()[:1].upper() in ("R", "C"):
                    rc_count += 1
                    if rc_count >= rc_threshold:
                        return True
        return False

    def _strip_pex_file(self, input_pex, output_path):
        """Drop short R/C parasitic lines, keep active devices."""
        if self.debug:
            print(f"  -> [Optimizer] Stripping: {os.path.basename(input_pex)}")

        with open(input_pex, "r") as f_in, open(output_path, "w") as f_out:
            for line in f_in:
                tokens = line.strip().split()
                if not tokens:
                    f_out.write(line)
                    continue
                first = tokens[0][0].upper()
                if first in (".", "*", "X", "D", "M", "Q", "J", "B",
                             "E", "F", "G", "H", "L", "V", "I"):
                    f_out.write(line)
                elif first in ("R", "C"):
                    if len(tokens) > 6:
                        f_out.write(line)
                else:
                    f_out.write(line)

    # ------------------------------------------------------------------
    # Netlist optimisation
    # ------------------------------------------------------------------

    def optimize_generator_netlist(self, original_tb_path):
        """
        Build generator_tb_optimized.sp:
          1. Replace PEX includes with stripped versions
          2. Strip ALL analysis lines so generator.sp only runs tran 1p 1p
        """
        if self.debug:
            print("[Auto-Detect] Scanning testbench topology...")

        with open(original_tb_path, "r") as f:
            lines = f.readlines()

        tb_dir       = os.path.dirname(os.path.abspath(original_tb_path))
        new_lines    = []
        pex_detected = False

        for line in lines:
            lower = line.strip().lower()

            if lower.startswith(".include") or lower.startswith(".inc"):
                parts = line.split()
                if len(parts) > 1:
                    raw     = parts[1].strip("\"'")
                    inc_abs = os.path.abspath(os.path.join(tb_dir, raw))

                    if "_stripped" in os.path.basename(inc_abs):
                        safe = Path(inc_abs).as_posix()
                        new_lines.append(f".include \"{safe}\"\n")
                        if self.debug:
                            print(f"  -> [Optimizer] Already stripped: "
                                  f"{os.path.basename(inc_abs)}")
                        continue

                    if self._is_pex_file(inc_abs):
                        pex_detected = True
                        if self.debug:
                            print(f"  -> [Alert] PEX detected: "
                                  f"{os.path.basename(inc_abs)}")

                        base, ext    = os.path.splitext(inc_abs)
                        stripped_abs = os.path.abspath(base + "_stripped" + ext)
                        stripped_safe = Path(stripped_abs).as_posix()

                        if not os.path.exists(stripped_abs):
                            self._strip_pex_file(inc_abs, stripped_abs)
                        elif self.debug:
                            print(f"  -> [Optimizer] Using cached: "
                                  f"{os.path.basename(stripped_abs)}")

                        new_lines.append(f".include \"{stripped_safe}\"\n")
                        continue

            new_lines.append(line)

        content = _clean_for_generator("".join(new_lines))

        fast_tb_path = os.path.join(self.workdir, "generator_tb_optimized.sp")
        with open(fast_tb_path, "w") as f:
            f.write(content)

        if self.debug:
            if pex_detected:
                print(f"  -> [Optimizer] Written: {fast_tb_path}")
            else:
                print("  -> [Status] Pre-layout schematic. No PEX stripping needed.")

        return fast_tb_path

    # ------------------------------------------------------------------
    # Feature discovery
    # ------------------------------------------------------------------

    def _extract_active_device_params(self, log_content):
        ACTIVE_PREFIXES = ("m.", "q.", "d.", "j.", "b.")
        STAT_PARAMS = {
            "delvto", "mulu0", "vth0", "vth",
            "toxe", "toxp", "toxm",
            "ndep", "nsd", "ngate",
            "dvt0", "dvt1", "dvt2",
            "eta0", "etab",
            "dsub", "minv",
            "ua", "ub", "uc",
            "vsat", "a0", "ags",
            "b0", "b1",
            "keta",
            "a1", "a2",
            "rdsw", "rdswmin",
            "prwg", "prwb",
            "wr",
            "wint", "lint",
            "dwg", "dwb",
            "voff", "voffl",
            "cit", "cdsc", "cdscd", "cdscb",
            "tox",
            "u0",
            "cgso", "cgdo", "cgbo",
            "weff", "leff",
            "pclm", "pdiblc1", "pdiblc2", "pdiblcb",
            "drout",
        }
        all_params, _ = self.parser.parse_log(log_content)
        active = [
            k for k in all_params
            if k.lstrip("@").lower().startswith(ACTIVE_PREFIXES)
            and k.split("[")[-1].rstrip("]").lower() in STAT_PARAMS
        ]

        if self.debug:
            print(f"   -> Total params from show : {len(all_params)}")
            print(f"   -> Active-device params   : {len(active)} (statistical only)")

        return active

    def discover_features(self):
        """
        Find which device params vary across MC seeds (true statistical X).
        """
        if self.active_params is not None:
            if self.debug:
                print("[Experiment] Discovery already done — skipping.")
            return

        # --- ngspice_mc flow ---
        if self.simulator_type == "ngspice_mc":
            if self.debug:
                print("[Experiment] Starting ngspice_mc Feature Discovery (mc_params_only mode)...")

            disc_dir = os.path.join(self.workdir, "discovery")
            mc_file  = self.simulator.run_mc_params_dump(5, output_dir=disc_dir)
            df_mc    = self.parser.parse_mc_params(mc_file)

            if df_mc.empty:
                raise RuntimeError(
                    "[Error] mc_params.mc is empty or unreadable after discovery run.\n"
                    f"Check {disc_dir}/mc_params_dump.log"
                )

            self.active_params = list(df_mc.columns)
            if self.debug:
                print(f"[Experiment] Discovery complete.")
                print(f"   -> True statistical inputs: {len(self.active_params)}")
            return

        # --- ngspice flow ---
        if self.debug:
            print("[Experiment] Starting Feature Discovery...")

        self.fast_generator_tb = self.optimize_generator_netlist(self.spfile)

        if self.debug:
            print("[Experiment] Running discovery (.op) on stripped TB...")

        stripped_sim = Simulator(self.fast_generator_tb, self.workdir, ngspice_cmd=self.ngspice_cmd)
        log_path     = stripped_sim.run_discovery(seed=0)

        with open(log_path) as f:
            log_content = f.read()

        all_param_names = self._extract_active_device_params(log_content)

        if not all_param_names:
            raise RuntimeError(
                "[Error] No active-device parameters found.\n"
                "Check that MOSFETs survived the PEX strip and that\n"
                f"'show all' output appears in: {log_path}"
            )

        if self.debug:
            print(f"[Experiment] Candidates to variance-check: {len(all_param_names)}")

        BATCH_SIZE = 200
        all_dfs    = []
        total      = (len(all_param_names) - 1) // BATCH_SIZE + 1

        for i in range(0, len(all_param_names), BATCH_SIZE):
            chunk     = all_param_names[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            if self.debug:
                print(f"   -> Batch {batch_num}/{total} ({len(chunk)} params)...")

            try:
                data_file = self.simulator.run_fast_generator(
                    chunk, n_samples=5, tb_path=self.fast_generator_tb
                )
                df = self.parser.parse_fast_output(data_file)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"      [Warning] Batch {batch_num} failed: {e}")

        if not all_dfs:
            raise RuntimeError(
                "All discovery batches failed!\n"
                f"Check generator.sp at: {self.workdir}\n"
                f"And stripped TB at: {self.fast_generator_tb}"
            )

        df_all  = pd.concat(all_dfs, axis=1)
        stds    = df_all.std()
        varying = stds[stds > 1e-20].index.tolist()

        seen = set()
        self.active_params = []
        for p in varying:
            if p not in seen:
                seen.add(p)
                self.active_params.append(p)

        if self.debug:
            print(f"[Experiment] Discovery complete.")
            print(f"   -> True statistical inputs: {len(self.active_params)}")

        if not self.active_params:
            raise RuntimeError(
                "[Error] Zero statistical parameters found.\n"
                "Check PDK MC models are activated in your testbench."
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def execute_training(self, seeds):
        if self.active_params is None:
            self.discover_features()

        # --- ngspice_mc flow ---
        if self.simulator_type == "ngspice_mc":
            seeds   = list(seeds)
            n_seeds = len(seeds)

            print(f"\n[Experiment] ngspice_mc: dumping {n_seeds} MC parameter rows (X)...")
            train_dir = os.path.join(self.workdir, "mc_params_training")
            mc_file   = self.simulator.run_mc_params_dump(n_seeds, output_dir=train_dir)
            df_mc     = self.parser.parse_mc_params(mc_file)
            df_mc     = df_mc.reset_index(drop=True)
            X_df      = df_mc.reindex(columns=self.active_params, fill_value=0.0)
            if self._mc_params_df is None:
                self._mc_params_df = df_mc

            print(f"[Experiment] ngspice_mc: running {n_seeds} full-circuit seeds in parallel "
                  f"({self.ngspice_np} workers) for Y...")
            train_sweep_dir = os.path.join(self.workdir, "mc_sweep_training")
            args_list = [
                (s + 1, self.spfile, train_sweep_dir, self.debug, self.ngspice_cmd)
                for s in seeds
            ]
            results_map = {}
            ex = self._get_mc_executor()
            futures = {ex.submit(_mc_worker, a): a[0] for a in args_list}
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                seed_k = futures[fut]
                try:
                    sk, targs = fut.result()
                    results_map[sk] = targs
                except Exception as e:
                    print(f"  [Experiment] seed_k={seed_k} exception: {e}")
                    results_map[seed_k] = {}
                if (i + 1) % 10 == 0:
                    print(f"  -> {i+1}/{n_seeds} complete")

            target_list = [results_map.get(s + 1, {}) for s in seeds]
            Y_df = pd.DataFrame(target_list)
            if len(Y_df) > 0:
                Y_df.insert(0, "Seed", seeds)

            return X_df, Y_df

        # --- ngspice flow ---
        seeds     = list(seeds)
        n_workers = min(self.ngspice_np, len(seeds))

        print(f"\n[Experiment] Launching {n_workers} workers for {len(seeds)} seeds...")

        args_list = [
            (s, self.spfile, self.workdir, self.active_params, self.debug, self.ngspice_cmd)
            for s in seeds
        ]

        results_map = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker, a): a[0] for a in args_list}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                seed_done = futures[future]
                try:
                    s, feats, targs = future.result()
                    results_map[s]  = (feats, targs)
                except Exception as e:
                    print(f"  [Experiment] Seed {seed_done} exception: {e}")
                    results_map[seed_done] = ({}, {})

                if self.debug and (i + 1) % 10 == 0:
                    print(f"  -> {i+1}/{len(seeds)} complete")

        X_data, Y_data = [], []
        for s in seeds:
            feats, targs = results_map.get(s, ({}, {}))
            X_data.append(feats)
            Y_data.append(targs)

        X_df = pd.DataFrame(X_data).reindex(columns=self.active_params, fill_value=0.0)
        Y_df = pd.DataFrame(Y_data)
        if len(Y_df) > 0 and len(seeds) == len(Y_df):
            Y_df.insert(0, "Seed", seeds)

        return X_df, Y_df

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def execute_generation(self, n_samples=10000, seed=None):
        if self.active_params is None:
            self.discover_features()

        # --- ngspice_mc flow ---
        if self.simulator_type == "ngspice_mc":
            if self.debug:
                print(f"[Experiment] ngspice_mc: dumping {n_samples:,} MC parameter rows...")

            gen_dir = os.path.join(self.workdir, "mc_params_generation")
            mc_file = self.simulator.run_mc_params_dump(n_samples, output_dir=gen_dir)
            df_mc   = self.parser.parse_mc_params(mc_file)
            df_mc   = df_mc.reset_index(drop=True)
            X_df    = df_mc.reindex(columns=self.active_params, fill_value=0.0)
            self._mc_params_df = X_df

            if self.debug:
                print(f"[Experiment] Generation complete. Shape: {X_df.shape}")

            return X_df

        # --- ngspice flow ---
        if self.fast_generator_tb is None:
            self.fast_generator_tb = self.optimize_generator_netlist(self.spfile)

        if self.debug:
            print(f"[Experiment] Generating {n_samples:,} virtual samples...")

        data_file = self.simulator.run_fast_generator(
            self.active_params, n_samples, tb_path=self.fast_generator_tb
        )

        X_df = self.parser.parse_fast_output(data_file)
        X_df = X_df.reindex(columns=self.active_params, fill_value=0.0)

        if self.debug:
            print(f"[Experiment] Generation complete. Shape: {X_df.shape}")

        return X_df

    # ------------------------------------------------------------------
    # Audit (Specific Points)
    # ------------------------------------------------------------------

    def execute_audit(self, input_df, audit_dir=None):
        """Simulate specific parameter points via monte_seed replay or alter."""
        if self.active_params is None:
            self.discover_features()

        # --- ngspice_mc flow ---
        if self.simulator_type == "ngspice_mc":
            if audit_dir is None:
                audit_dir = os.path.join(self.workdir, "al_audits")
            os.makedirs(audit_dir, exist_ok=True)

            print(f"\n[Experiment] ngspice_mc: auditing {len(input_df)} points via monte_seed replay...")

            pool_positions = list(input_df.index)
            seed_k_list    = [p + 2 for p in pool_positions]

            args_list = [
                (sk, self.spfile, self.workdir, self.debug, self.ngspice_cmd)
                for sk in seed_k_list
            ]

            results_map = {}
            ex = self._get_mc_executor()
            futures = {ex.submit(_mc_worker, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(futures):
                sk = futures[future]
                try:
                    sk_done, targets = future.result()
                    results_map[sk_done] = targets
                except Exception as e:
                    print(f"  [Audit] seed_k={sk} exception: {e}")
                    results_map[sk] = {}

            if self._mc_params_df is not None:
                X_rows = [
                    self._mc_params_df.iloc[p].to_dict() if p < len(self._mc_params_df)
                    else {}
                    for p in pool_positions
                ]
            else:
                X_rows = [input_df.loc[p].to_dict() for p in pool_positions]

            Y_data = [results_map.get(sk, {}) for sk in seed_k_list]
            X_df = pd.DataFrame(X_rows).reindex(columns=self.active_params, fill_value=0.0)
            Y_df = pd.DataFrame(Y_data)
            return X_df, Y_df

        # --- ngspice flow ---
        if audit_dir is None:
            audit_dir = os.path.join(self.workdir, "audits")
        os.makedirs(audit_dir, exist_ok=True)

        print(f"\n[Experiment] Auditing {len(input_df)} extreme outlier points...")
        X_data, Y_data = [], []

        for i, row in input_df.iterrows():
            audit_tb = os.path.join(audit_dir, f"worst_case_rank_{i+1}.sp")
            with open(self.spfile, "r") as f:
                content = f.read()

            alter_commands = "\n"
            for col in input_df.columns:
                try:
                    clean_val = float(row[col])
                except (ValueError, TypeError):
                    continue
                name = col if col.startswith("@") else f"@{col}"
                alter_commands += f"    alter {name} = {clean_val}\n"

            content = _CONTROL_RE.sub("", content)

            audit_control = (
                "\n.control\n"
                "  set noaskquit\n"
                "  set nomarch\n"
                "  set noisy = 0\n"
                f"{alter_commands}\n"
                "  save all\n"
                "  run\n"
                "  show all\n"
                "  quit\n"
                ".endc\n"
            )
            content = _END_RE.sub(f"{audit_control}\n.end", content)

            with open(audit_tb, "w") as f:
                f.write(content)

            audit_log_save = os.path.join(audit_dir, f"worst_case_rank_{i+1}.log")
            import subprocess
            with open(audit_log_save, "w") as out:
                subprocess.run([self.ngspice_cmd, "-b", audit_tb], stdout=out, stderr=out)

            with open(audit_log_save, "r") as f:
                feats, targs = self.parser.parse_log(f.read(), relevant_params=self.active_params)
                X_data.append(feats)
                Y_data.append(targs)

            print(f"    [Audit] Saved verification netlist: {audit_tb}")
            print(f"    [Audit] Saved verification log:     {audit_log_save}")

        X_df = pd.DataFrame(X_data).reindex(columns=self.active_params, fill_value=0.0)
        Y_df = pd.DataFrame(Y_data)
        return X_df, Y_df
