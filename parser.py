import re
import pandas as pd


# ---------------------------------------------------------------------------
# ngspice internal variable names that must never be treated as .measure
# results — extend if you see more bleed-through in your logs.
# ---------------------------------------------------------------------------
_NGSPICE_INTERNALS = {
    "version", "temperature", "temp", "tnom", "gmin", "reltol", "abstol",
    "vntol", "chgtol", "pivot", "itl1", "itl2", "itl4", "itl5",
    "method", "maxord", "tstep", "tstop", "tstart",
    "m", "rows", "available", "size", "value", "trig", "targ", "at", 
    "failed", "run", "loop_idx", "myseed", "seed", "initial", "id", "param"
}


class NgspiceParser:
    def __init__(self, debug=False):
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_log(self, logtext, relevant_params=None):
        """
        Parse a standard ngspice log file.

        Returns:
            features (dict)  X variables  keyed as "@device[param]"
            targets  (dict)  Y variables  keyed by .measure result name
        """
        features = self._parse_features(logtext, relevant_params)
        targets  = self._parse_targets(logtext)
        return features, targets

    def parse_mc_params(self, filepath):
        """
        Parse mc_params.mc produced by ngspice mc_params_only=1 mode.

        File format (space-delimited):
            run  param_0  param_1  ...  param_N-1
            1    0        0        ...  0           <- nominal row (skipped)
            2    draw_0   draw_1   ...  draw_N-1    <- seed_index 2
            3    draw_0   draw_1   ...  draw_N-1    <- seed_index 3
            ...

        Returns a pd.DataFrame with seed_index as integer index (starting at 2).
        """
        try:
            df = pd.read_csv(filepath, sep=r"\s+", engine="python",
                             header=0, index_col=0)
            df.index.name = "seed_index"
            # seed_index starts at 2 (no nominal row); drop index=1 is a no-op
            # but kept for safety in case an old-format file is passed.
            df = df.drop(index=1, errors="ignore")
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(how="all")
            return df
        except Exception as e:
            print(f"[Parser Error] Failed to parse mc_params '{filepath}': {e}")
            return pd.DataFrame()

    def parse_mc_sweep_targets(self, logtext):
        """
        Parse N sets of .measure results from a sweep monte=N log.
        Returns a list of target dicts (one per MC iteration), in order.
        """
        measure_line_re = re.compile(
            r"([a-zA-Z][a-zA-Z0-9_]*)\s*[=:]\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b",
            re.IGNORECASE
        )
        sections = re.findall(
            r"Measurements for .+?(?=\nMeasurements for |\Z)",
            logtext, re.DOTALL | re.IGNORECASE,
        )
        results = []
        for section in sections:
            targets = {}
            for name, val in measure_line_re.findall(section):
                if name.lower() in _NGSPICE_INTERNALS:
                    continue
                try:
                    targets[name] = float(val)
                except ValueError:
                    pass
            results.append(targets)
        if self.debug and not results:
            print("[Parser] No 'Measurements for' sections found in sweep log.")
        return results

    def parse_fast_output(self, filepath):
        """
        Parse the space-separated file produced by run_fast_generator().

        Returns:
            pd.DataFrame — columns are parameter names, rows are samples.
        """
        try:
            df = pd.read_csv(filepath, sep=r"\s+", engine="python")

            if "run" in df.columns:
                df = df.drop(columns=["run"])

            df = df.dropna(axis=1, how="all")
            df = df.apply(pd.to_numeric, errors="coerce")

            n_before  = len(df)
            df        = df.dropna()
            n_dropped = n_before - len(df)

            if n_dropped > 0:
                print(f"[Parser] Dropped {n_dropped} malformed rows.")

            return df

        except Exception as e:
            print(f"[Parser Error] Failed to parse '{filepath}': {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_features(self, logtext, relevant_params=None):
        """
        Extract device parameters from the ngspice log.
        """
        features = {}
        # Use lowercase for case-insensitive matching
        keep_set = {p.lower() for p in relevant_params} if relevant_params else None

        # --- Format B: @device[param] = value ---
        print_re = re.compile(
            r"(@[a-zA-Z0-9_.]+\[[a-zA-Z0-9_]+\])\s*=\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        for match in print_re.finditer(logtext):
            key = match.group(1).lower()
            if keep_set and key not in keep_set:
                continue
            try:
                features[key] = float(match.group(2))
            except ValueError:
                pass

        if features:
            return features

        # --- Format A: show table fallback ---
        lines    = logtext.splitlines()
        devices  = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_table:
                    devices  = []
                    in_table = False
                continue

            parts = stripped.split()

            if parts[0].lower() == "device":
                devices  = [d.lower() for d in parts[1:]]
                in_table = True
                continue

            if in_table and devices and len(parts) == len(devices) + 1:
                param_name = parts[0].lower()
                for dev, val in zip(devices, parts[1:]):
                    key = f"@{dev}[{param_name}]".lower()
                    if keep_set and key not in keep_set:
                        continue
                    try:
                        features[key] = float(val)
                    except ValueError:
                        pass

        if self.debug and not features and relevant_params:
            print("[Parser Warning] No device features found.")

        return features

    def _parse_targets(self, logtext):
        """
        Extract .measure results from the log.

        Tries section-aware extraction first (anchored to
        'Measurements for ...' headers), then falls back to a global scan.
        """
        targets = {}

        # Strategy B: permissive global scan
        # Matches "name = value" or "name : value" anywhere on a line
        measure_line_re = re.compile(
            r"([a-zA-Z][a-zA-Z0-9_]*)\s*[=:]\s*"
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b",
            re.IGNORECASE
        )

        # Strategy A: section-aware
        sections = re.findall(
            r"Measurements for .+?(?=\nMeasurements for |\Z)",
            logtext,
            re.DOTALL | re.IGNORECASE,
        )

        if sections:
            for section in sections:
                for name, val in measure_line_re.findall(section):
                    if name.lower() in _NGSPICE_INTERNALS:
                        continue
                    try:
                        targets[name] = float(val)
                    except ValueError:
                        pass
        else:
            # Strategy B: global scan fallback
            if self.debug:
                print("[Parser] No 'Measurements for' section — global scan.")
            for name, val in measure_line_re.findall(logtext):
                if name.lower() in _NGSPICE_INTERNALS:
                    continue
                try:
                    targets[name] = float(val)
                except ValueError:
                    pass

        if self.debug and not targets:
            print("[Parser Warning] No .measure results found.")

        return targets