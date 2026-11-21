import subprocess
import os
import secrets
import shutil
import re


_ANALYSIS_RE = re.compile(
    r"^\s*\.(tran|ac|dc|op|measure)\b.*$",
    re.MULTILINE | re.IGNORECASE,
)
# Matches only simulation analysis cards (not .measure) for sweep injection
_SIM_ANALYSIS_RE = re.compile(
    r"^(\s*\.(tran|ac|dc|op)\b.*)$",
    re.MULTILINE | re.IGNORECASE,
)
_CONTROL_RE = re.compile(
    r"^\s*\.control\b[\s\S]*?\.endc",
    re.MULTILINE | re.IGNORECASE
)
_END_RE = re.compile(
    r"^\s*\.end\b.*",
    re.MULTILINE | re.IGNORECASE
)
_OUTPUT_RE = re.compile(
    r"^\s*\.(print|plot|probe)\b.*$",
    re.MULTILINE | re.IGNORECASE,
)

def _gen_worker_task(param_list, n_samples, tb_path, start_idx, data_file, ngspice_cmd="ngspice"):
    """Standalone worker for cross-platform process-pool compatibility"""
    with open(tb_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean the netlist
    content = _CONTROL_RE.sub("", content)
    content = _ANALYSIS_RE.sub("", content)
    content = _END_RE.sub("", content)

    SEED_OFFSET = int(10007 + start_idx)
    n_samples_int = int(n_samples)
    _tok = secrets.token_hex(4)
    gen_file = data_file.replace(".txt", f"_{_tok}.sp")
    tmp_log = data_file + f"_{_tok}.tmp"
    
    if os.path.exists(tmp_log):
        os.remove(tmp_log)

    script = (
        content + "\n"
        ".control\n"
        "  set noaskquit\n"
        "  set nomarch\n"
        "  set noisy = 0\n"
        "\n"
        f"  let n_samples = {n_samples_int}\n"
        "  let loop_idx = 0\n"
        "\n"
        "  while loop_idx < n_samples\n"
        "\n"
        f"    let myseed = {SEED_OFFSET} + loop_idx\n"
        "    set seed = $&myseed\n"
        "    reset\n"
        "    op\n"
        f'    echo "--- RUN $&loop_idx ---"\n'
        "    show all\n"
        "    let loop_idx = loop_idx + 1\n"
        "  end\n"
        "\n"
        "  quit\n"
        ".endc\n"
        "\n"
        ".end\n"
    )
    
    try:
        with open(gen_file, "w", encoding="utf-8") as f:
            f.write(script)

        with open(tmp_log, "w", encoding="utf-8") as f_out:
            subprocess.run([ngspice_cmd, "-b", gen_file], stdout=f_out, stderr=subprocess.DEVNULL)
        
        # Parse the tmp_log and write out the required fast_data CSV format
        if os.path.exists(tmp_log):
            with open(tmp_log, "r", encoding="utf-8", errors="replace") as f:
                logtext = f.read()
                
            runs = logtext.split("--- RUN ")
            
            from parser import NgspiceParser
            p = NgspiceParser()
            
            with open(data_file, "w", encoding="utf-8") as fout:
                fout.write("run " + " ".join(param_list) + "\n")
                for run_block in runs[1:]:
                    if not run_block.strip(): continue
                    lines = run_block.strip().split("\n")
                    run_idx = lines[0].strip().split(" ")[0]
                    
                    # Use the robust parser from the core engine
                    features = p._parse_features(run_block, relevant_params=param_list)
                    
                    row = [run_idx]
                    for param in param_list:
                        val = features.get(param.lower(), 0.0)
                        row.append(str(val))
                    fout.write(" ".join(row) + "\n")
        else:
            # Fallback if ngspice failed entirely
            with open(data_file, "w", encoding="utf-8") as fout:
                fout.write("run " + " ".join(param_list) + "\n")
    finally:
        if os.path.exists(tmp_log):
            try: os.remove(tmp_log)
            except OSError: pass
        if os.path.exists(gen_file):
            try: os.remove(gen_file)
            except OSError: pass

    return data_file



class Simulator:
    def __init__(self, spfile, workdir="./temp", ngspice_cmd="ngspice"):
        self.spfile      = spfile
        self.workdir     = workdir
        self.ngspice_cmd = ngspice_cmd
        os.makedirs(workdir, exist_ok=True)

    def _base_clean(self, content):
        """Strip .control blocks and .end only."""
        content = re.sub(r"^\s*\.control\b[\s\S]*?\.endc", "",
                         content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r"^\s*\.end\b.*", "",
                         content, flags=re.MULTILINE | re.IGNORECASE)
        return content

    def run(self, seed=None):
        """
        Full simulation for training data collection (X and Y).
        Keeps .tran and .measure so Y is produced.
        Uses 'show all' to descend into subcircuit wrappers like nfet_06v0.
        """
        seed_val = seed if seed is not None else 0
        seed_dir = os.path.join(self.workdir, f"seed_{seed_val}")
        os.makedirs(seed_dir, exist_ok=True)

        tmpfile = os.path.join(seed_dir, f"run_{seed_val}.sp")
        logfile = os.path.join(seed_dir, f"run_{seed_val}.log")

        if os.path.exists(logfile) and os.path.getsize(logfile) > 0:
            return logfile

        shutil.copyfile(self.spfile, tmpfile)

        with open(tmpfile, "r") as f:
            content = f.read()

        content = self._base_clean(content)

        seed_val_str = str(seed_val)
        control_block = (
            "\n.control\n"
            "  set noaskquit\n"
            "  set nomarch\n"
            "  set noisy = 0\n"
            "  set seed = " + seed_val_str + "\n"
            "\n"
            "  save all\n"
            "  run\n"
            "\n"
            "  show all\n"
            "\n"
            ".endc\n"
            "\n"
            ".end\n"
        )

        with open(tmpfile, "w") as f:
            f.write(content + control_block)

        with open(logfile, "w") as out:
            result = subprocess.run(
                [self.ngspice_cmd, "-b", tmpfile],
                stdout=out, stderr=out,
            )

        if result.returncode != 0:
            print(f"[Simulator Warning] ngspice exited code {result.returncode} "
                  f"for seed={seed_val}. Check {logfile}")

        return logfile

    def run_discovery(self, seed=0):
        """
        Discovery run: replaces all analysis with .op so it completes
        in milliseconds. Only needs device param names — no Y required.
        """
        disc_dir = os.path.join(self.workdir, "discovery")
        os.makedirs(disc_dir, exist_ok=True)

        tmpfile = os.path.join(disc_dir, "discovery.sp")
        logfile = os.path.join(disc_dir, "discovery.log")

        shutil.copyfile(self.spfile, tmpfile)

        with open(tmpfile, "r") as f:
            content = f.read()

        content = self._base_clean(content)
        content = _ANALYSIS_RE.sub("", content)
        content += "\n.op\n"

        control_block = (
            "\n.control\n"
            "  set noaskquit\n"
            "  set nomarch\n"
            "  set noisy = 0\n"
            "  set seed = " + str(seed) + "\n"
            "\n"
            "  save all\n"
            "  run\n"
            "\n"
            "  show all\n"
            "\n"
            ".endc\n"
            "\n"
            ".end\n"
        )

        with open(tmpfile, "w") as f:
            f.write(content + control_block)

        with open(logfile, "w") as out:
            result = subprocess.run(
                [self.ngspice_cmd, "-b", tmpfile],
                stdout=out, stderr=out,
            )

        if result.returncode != 0:
            print(f"[Simulator Warning] Discovery run exited code "
                  f"{result.returncode}. Check {logfile}")

        return logfile

    def run_mc_params_dump(self, n_samples, output_dir=None):
        """
        Run ngspice in mc_params_only=1 mode to dump raw N(0,1) sigma draws
        without a circuit solve. Returns path to mc_params.mc.

        The file has n_samples actual MC rows (rows 2..n_samples+1; row 1 is
        the nominal zeros row). Row k in the file corresponds to monte_seed=k.
        """
        if output_dir is None:
            output_dir = os.path.join(self.workdir, "mc_params_dump")
        os.makedirs(output_dir, exist_ok=True)

        tmpfile = os.path.join(output_dir, "mc_params_dump.sp")
        mc_file = os.path.join(output_dir, "mc_params.mc")
        logfile = os.path.join(output_dir, "mc_params_dump.log")

        with open(self.spfile, "r") as f:
            content = f.read()

        content = self._base_clean(content)
        # Replace analysis cards with a cheap .op; mc_params_only=1 skips the
        # circuit solve — the analysis type only triggers the parameter scan.
        content = _ANALYSIS_RE.sub("", content)

        # sweep monte=N gives N actual rows (seeds 2..N+1), aligned with the
        # N iterations of a full-circuit sweep monte=N run.
        n_sweep = n_samples
        injected = (
            f"\n.op sweep monte={n_sweep}\n"
            "\n.options mc_params_only=1\n"
            "\n.control\n"
            "  set noaskquit\n"
            "  set nomarch\n"
            "  set noisy = 0\n"
            "  run\n"
            "  quit\n"
            ".endc\n"
            "\n.end\n"
        )

        with open(tmpfile, "w") as f:
            f.write(content + injected)

        with open(logfile, "w") as out:
            result = subprocess.run(
                [self.ngspice_cmd, "-b", tmpfile],
                stdout=out, stderr=out,
                cwd=output_dir,
            )

        if result.returncode != 0:
            print(f"[Simulator Warning] mc_params_dump exited code "
                  f"{result.returncode}. Check {logfile}")

        if not os.path.exists(mc_file):
            raise RuntimeError(
                f"[Simulator] mc_params.mc not created. Check {logfile}"
            )

        return mc_file

    def run_monte_seed(self, seed_k, output_dir=None):
        """
        Full circuit simulation with monte_seed=seed_k, reproducing the exact
        MC parameter draw from row seed_k of mc_params.mc. Returns logfile path.
        """
        if output_dir is None:
            output_dir = os.path.join(self.workdir, f"mc_seed_{seed_k}")
        os.makedirs(output_dir, exist_ok=True)

        tmpfile = os.path.join(output_dir, f"mc_seed_{seed_k}.sp")
        logfile = os.path.join(output_dir, f"mc_seed_{seed_k}.log")

        if os.path.exists(logfile) and os.path.getsize(logfile) > 0:
            return logfile

        with open(self.spfile, "r") as f:
            content = f.read()

        content = self._base_clean(content)
        content = _OUTPUT_RE.sub("", content)

        meas_file = os.path.join(output_dir, "measurements.mt0")

        control_block = (
            f"\n.options monte_seed={seed_k} NOINIT NOREFVALUE\n"
            "\n.control\n"
            "  set noaskquit\n"
            "  set nomarch\n"
            "  set noisy = 0\n"
            "  set measoutfile = \"measurements.mt0\"\n"
            "  run\n"
            ".endc\n"
            "\n.end\n"
        )

        with open(tmpfile, "w") as f:
            f.write(content + control_block)

        with open(logfile, "w") as out:
            result = subprocess.run(
                [self.ngspice_cmd, "-b", tmpfile],
                stdout=out, stderr=out,
                cwd=output_dir,
            )

        if result.returncode != 0:
            print(f"[Simulator Warning] monte_seed={seed_k} exited code "
                  f"{result.returncode}. Check {logfile}")

        if os.path.exists(meas_file):
            return meas_file
        return logfile

    def run_mc_sweep(self, n_samples, output_dir=None):
        """
        Full-circuit MC sweep with sweep monte=n_samples using aligned
        deterministic seeding.  Iteration k (0-indexed) uses seed k+2,
        matching row k+2 of mc_params.mc from run_mc_params_dump(n_samples).

        Returns path to the combined logfile containing n_samples sets of
        .measure results, one per iteration.
        """
        if output_dir is None:
            output_dir = os.path.join(self.workdir, "mc_sweep")
        os.makedirs(output_dir, exist_ok=True)

        tmpfile = os.path.join(output_dir, "mc_sweep.sp")
        logfile = os.path.join(output_dir, "mc_sweep.log")

        with open(self.spfile, "r") as f:
            content = f.read()

        content = self._base_clean(content)
        content = _OUTPUT_RE.sub("", content)

        # Append "sweep monte=N" to the first simulation analysis card
        _injected = [False]

        def _append_sweep(m):
            if _injected[0]:
                return m.group(0)
            line = m.group(0).rstrip()
            if not re.search(r'\bsweep\s+monte\b', line, re.IGNORECASE):
                line = line + f" sweep monte={n_samples}"
            _injected[0] = True
            return line + "\n"

        content = _SIM_ANALYSIS_RE.sub(_append_sweep, content)
        if not _injected[0]:
            content += f"\n.op sweep monte={n_samples}\n"

        meas_file = os.path.join(output_dir, "measurements.mt0")

        control_block = (
            "\n.options NOINIT NOREFVALUE\n"
            "\n.control\n"
            "  set noaskquit\n"
            "  set nomarch\n"
            "  set noisy = 0\n"
            "  set measoutfile = \"measurements.mt0\"\n"
            "  run\n"
            ".endc\n"
            "\n.end\n"
        )

        with open(tmpfile, "w") as f:
            f.write(content + control_block)

        with open(logfile, "w") as out:
            result = subprocess.run(
                [self.ngspice_cmd, "-b", tmpfile],
                stdout=out, stderr=out,
                cwd=output_dir,
            )

        if result.returncode != 0:
            print(f"[Simulator Warning] mc_sweep exited code {result.returncode}. "
                  f"Check {logfile}")

        if os.path.exists(meas_file):
            return meas_file

        # measoutfile not written — fall back to log (older ngspice builds)
        print("[Simulator] measoutfile not created; falling back to log parsing.")
        return logfile

    def run_fast_generator(self, param_list, n_samples=10000, tb_path=None):
        import concurrent.futures
        import multiprocessing
        import sys

        if tb_path is None:
            tb_path = self.spfile

        out_file = os.path.join(self.workdir, "X_fast_data.txt")
        
        # Split into smaller chunks for better progress visibility
        n_workers = multiprocessing.cpu_count()
        chunk_size = 50  # 50 samples per chunk
        tasks = []
        
        start_idx = 0
        chunk_id = 0
        while start_idx < n_samples:
            count = min(chunk_size, n_samples - start_idx)
            temp_out = os.path.join(self.workdir, f"X_fast_part_{chunk_id}.txt")
            tasks.append((param_list, count, tb_path, start_idx, temp_out, self.ngspice_cmd))
            start_idx += count
            chunk_id += 1

        print(f"[Simulator] Spawning {n_workers} workers to process {n_samples} samples in {len(tasks)} chunks...")
        
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_count = {executor.submit(_gen_worker_task, *t): t[1] for t in tasks}
            
            for future in concurrent.futures.as_completed(future_to_count):
                count = future_to_count[future]
                completed += count
                percent = (completed / n_samples) * 100
                sys.stdout.write(f"\r  -> Progress: {completed}/{n_samples} samples ({percent:.1f}%)   ")
                sys.stdout.flush()
            print() # Newline after progress bar

        # Merge parts
        print(f"  -> Merging {len(tasks)} parts into {out_file}...")
        with open(out_file, "w") as fout:
            for i in range(len(tasks)):
                part_file = os.path.join(self.workdir, f"X_fast_part_{i}.txt")
                if not os.path.exists(part_file): continue
                with open(part_file, "r") as fin:
                    lines = fin.readlines()
                    if i == 0:
                        # Keep header only for the first file
                        fout.writelines(lines)
                    else:
                        # Skip header for subsequent files
                        fout.writelines(lines[1:])
                os.remove(part_file)

        return out_file

