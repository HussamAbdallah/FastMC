# ngspice Internal Modifications

A detailed walkthrough of every layer touched inside ngspice-47, following the data flow from netlist to output.

---

## How ngspice processes a simulation (the baseline)

Before explaining the changes, it helps to understand the normal path:

1. The **netlist parser** (`inpcom.c`, `inp2dot.c`) reads the `.sp` file line by line, expanding `.param` definitions, registering device instances, and building analysis cards (`.tran`, `.ac`, etc.)
2. Each analysis card becomes a `JOB` structure, and the global simulation state lives in a `TSKtask` struct defined in `tskdefs.h`
3. The **option system** (`optdefs.h` → `cktsopt.c`) maps `.options` keywords to integer enum values, then writes them into `TSKtask` fields via `CKTsetOpt()`
4. The **expression evaluator** (`xpressn.c`) handles built-in functions like `agauss(nominal, av, sigma)` at circuit-solve time — every time a device parameter references agauss, this function is called to draw a Gaussian random number
5. The **run commander** (`runcoms.c`) orchestrates the actual simulation loops — it calls the circuit solver, advances time steps, and handles sweep iterations
6. The **output interface** (`outitf.c`) handles everything printed to the console and log file during a run

We inserted new behaviour at every one of these layers.

---

## Layer 1 — `tskdefs.h`: adding three new fields to `TSKtask`

```c
/* Monte Carlo sweep options (set by "sweep monte=N" on analysis cards) */
int TSKmonteN;               /* number of MC iterations (0 or 1 = disabled) */
unsigned int TSKmcParamsOnly:1; /* skip full solver; run .op + dump params to .mc file */
int TSKmonteSeed;            /* >0: reproduce this seed index for a single full sim */
```

`TSKtask` is the struct that travels with the simulation from parse time all the way through the solver. By adding fields here, every subsystem that receives a `task*` pointer can query these flags without any additional plumbing. This is the idiomatic ngspice way to add simulation-wide state.

- `TSKmonteN` — how many Monte Carlo iterations to run in the fast parameter-dump mode
- `TSKmcParamsOnly` — a 1-bit flag (bitfield) that tells the run loop to skip the full transient/AC solve and only collect agauss draws
- `TSKmonteSeed` — when > 0, tells the run loop to replay one specific seed index with a full circuit solve

---

## Layer 2 — `optdefs.h` + `cktsopt.c`: wiring `.options` keywords

```c
/* optdefs.h */
OPT_MC_PARAMS_ONLY = 300,  /* dump MC parameter vectors, skip circuit solver */
OPT_MONTE_SEED     = 301,  /* seed index to reproduce for a single full simulation */
```

ngspice's option system works as a table lookup: each `.options` keyword has an integer enum value, and `CKTsetOpt()` in `cktsopt.c` does a switch/case on that integer to write the value into the right `TSKtask` field.

```c
/* cktsopt.c */
case OPT_MC_PARAMS_ONLY:
    task->TSKmcParamsOnly = (val->iValue != 0);
    break;
case OPT_MONTE_SEED:
    task->TSKmonteSeed = val->iValue;
    break;
```

This is how a user-facing `.options mc_params_only=1` or `.options monte_seed=42` in the netlist gets turned into a C flag that the solver reads. Without the enum entries in `optdefs.h`, the keyword would be silently ignored. Without the switch cases in `cktsopt.c`, it would be recognized but never stored.

---

## Layer 3 — `inp2dot.c`: parsing `sweep monte=N` off analysis cards

```c
/* Strip "sweep monte=N" from the card line before any dot_* parser sees it.
 * Works for .tran, .ac, .dc, .op, and any other analysis type. */
{
    char *p = strstr(current->line, "sweep");
    if (p) {
        char *q = p + 5;
        while (*q == ' ' || *q == '\t') q++;
        if (strncmp(q, "monte=", 6) == 0) {
            int n = atoi(q + 6);
            if (n > 0 && task) {
                task->TSKmonteN = n;
                /* Remove "sweep monte=NNN" span from the line in-place */
                char *end = q + 6;
                while (*end && *end != ' ' && *end != '\t' && *end != '\n') end++;
                memmove(p, end, strlen(end) + 1);
            }
        }
    }
}
```

`INP2dot` is the function that handles all dot-command analysis cards (`.tran 10n 100u`, `.ac dec 10 1k 1g`, etc.). It runs before the type-specific parsers (`INPdomodel`, `INPdotran`, etc.) see the line.

The key design decision here: we intercept the line **before** the standard parser, extract the `monte=N` count, write it into `task->TSKmonteN`, and then **remove it from the line in-place** using `memmove`. This means the downstream `.tran` parser never sees `sweep monte=100` — it just sees `.tran 10n 100u` as normal. We didn't have to modify any of the individual analysis-type parsers.

---

## Layer 4 — `inpcom.c`: registering agauss parameter names before macro expansion

```c
/* Register agauss funcname for MC column naming (captures param name before
 * inp_expand_macros_in_deck inlines it away into agauss(nominal,av,s)). */
if (funcname && funcbody &&
    (strstr(funcbody, "agauss") || strstr(funcbody, "AGAUSS"))) {
    const char *p = strchr(funcbody, '(');
    if (p) {
        char tmp[256];
        strncpy(tmp, p + 1, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        for (char *q = tmp; *q && *q != ')'; q++)
            if (*q == ',') *q = ' ';
        double nom = 0.0, av = 0.0, sig = 0.0;
        if (sscanf(tmp, "%lf %lf %lf", &nom, &av, &sig) == 3)
            mc_agauss_register_func(funcname, av, sig);
    }
}
```

This is in `inp_new_func`, which processes `.param` lines like `.param Vth0 = agauss(0.42, 0.42, 3)`.

The problem: ngspice's macro expander (`inp_expand_macros_in_deck`) runs after `inp_new_func` and inlines every `.param` definition at every use site. By the time `xpressn.c` evaluates `agauss(0.42, 0.42, 3)`, the name `Vth0` is gone — there's no way to know which parameter name produced that draw.

Our hook runs **inside `inp_new_func` before the macro expansion**, when both the name (`funcname = "Vth0"`) and the body (`funcbody = "agauss(0.42,0.42,3)"`) are still together. We parse the av and sigma, then call `mc_agauss_register_func("Vth0", 0.42, 3)` to store the mapping. This table is later used to name the columns in the `.mc` output file.

---

## Layer 5 — `xpressn.c`: capturing and replaying agauss draws

This is the core of the whole mechanism. `xpressn.c` contains the expression evaluator — when ngspice encounters `agauss(nominal, av, sigma)` in a device parameter, it calls the `agauss` function here.

We added three things:

### A. Capture mode (during `mc_params_only` sweep)

During the parameter dump sweep, every `agauss()` call writes its raw N(0,1) draw into a global capture array instead of (or in addition to) returning the perturbed value. At the end of each MC iteration, the array is flushed as one row to the `.mc` output file, together with the parameter names from the registration table (Layer 4). The result is a file where each row is one complete MC parameter vector, in the same order agauss is evaluated.

### B. `mc_agauss_redraw_seeded(k)` — a new C function

```c
void mc_agauss_redraw_seeded(int k) {
    srand(k);
    TausSeed();           /* seed ngspice's Tausworthe RNG from srand state */
    for (int i = 0; i < TSKmonteN_captured; i++)
        gauss1();         /* advance RNG by exactly N draws */
}
```

This resets the RNG to a fully deterministic state for seed index `k`. `srand(k)` sets the C stdlib RNG, `TausSeed()` uses that to initialize ngspice's internal Tausworthe generator, and then N `gauss1()` calls advance it past the same draws that agauss would make for a fresh iteration — so the next call to `agauss()` produces exactly the same values as iteration `k` did in the original sweep.

### C. Replay mode (during `monte_seed` run)

When `TSKmonteSeed > 0`, before the circuit solve starts, we call `mc_agauss_redraw_seeded(TSKmonteSeed)`. From that point, every `agauss()` call in the circuit draws from the pre-positioned RNG — producing exactly the same parameter set as row `TSKmonteSeed` of the `.mc` file. The full transient/AC solve then runs on those parameters.

---

## Layer 6 — `runcoms.c`: the outer MC loop

`runcoms.c` handles `.control` block commands and orchestrates analysis runs. We added the loop that drives the two modes:

**mc_params_only loop** (for X generation):
```
for mc_iter = 0 to TSKmonteN-1:
    mc_agauss_redraw_seeded(mc_iter + 2)   ← seed 2..N+1
    run .op only (no transient)
    flush captured agauss draws to mc_params.mc
```

The `+ 2` offset reserves seed 0 and 1: seed 0 is the C stdlib default (undefined behaviour territory), seed 1 produces the nominal draw (all zeros after `TausSeed` initialization), so the first statistically independent variation starts at seed 2.

**monte_seed single run** (for Y generation / audit):
```
mc_agauss_redraw_seeded(TSKmonteSeed)
run full transient/AC/DC as normal
write .measure results to measoutfile
```

---

## Layer 7 — `outitf.c` + `options.c` / `spiceif.c`: suppressing noise output

ngspice prints several lines during every run that are useless and expensive to filter in Python:

- **"Reference value: ..."** — printed at the start of every sweep point
- **"Initial OP ..."** — printed before the DC operating point solve

We added:
- `ft_norefprint` global flag in `outitf.c` — when set, the reference value print is skipped
- `.options norefvalue` wired through `options.c` and `spiceif.c` → sets `ft_norefprint`
- `NOINIT` option plumbed similarly to suppress the Initial OP message
- `.set measoutfile = "measurements.mt0"` support so `.measure` results go to a compact separate file instead of stdout

The combined effect: a `monte_seed` run produces only the `.measure` outputs in `measurements.mt0` — no console noise for Python to strip.

---

## The complete data flow

```
.sp netlist
    │
    ├─ inp2dot.c:        extract "sweep monte=N" → TSKmonteN
    ├─ inpcom.c:         register agauss param names (Vth0 → av, sigma)
    ├─ optdefs.h /       .options mc_params_only=1 → TSKmcParamsOnly
    │  cktsopt.c:        .options monte_seed=K    → TSKmonteSeed
    │
    └─ runcoms.c:
         if mc_params_only:
             loop K=2..N+1:
                 xpressn.c: mc_agauss_redraw_seeded(K)
                 → .op only (no full solve)
                 → flush row to mc_params.mc          ← X matrix
         if monte_seed=K:
             xpressn.c: mc_agauss_redraw_seeded(K)
             → full transient solve
             → outitf.c: suppressed noise
             → measurements.mt0                       ← Y scalar
```

The two paths are **aligned by construction**: row `i` of `mc_params.mc` was produced by `mc_agauss_redraw_seeded(i + 2)`, and a `monte_seed=i+2` full run uses the identical RNG state — so X and Y are guaranteed to describe the same device corner.

---

## Files changed at a glance

| File | Location | What changed |
|------|----------|-------------|
| `tskdefs.h` | `src/include/ngspice/` | Added `TSKmonteN`, `TSKmcParamsOnly`, `TSKmonteSeed` to `TSKtask` struct |
| `optdefs.h` | `src/include/ngspice/` | Added `OPT_MC_PARAMS_ONLY = 300`, `OPT_MONTE_SEED = 301` |
| `cktsopt.c` | `src/spicelib/analysis/` | Added switch cases for the two new options |
| `inp2dot.c` | `src/spicelib/parser/` | Strip `sweep monte=N` from analysis cards; populate `TSKmonteN` |
| `inpcom.c` | `src/frontend/` | Hook in `inp_new_func` to register agauss param names before macro expansion |
| `xpressn.c` | `src/frontend/` | Capture mode, `mc_agauss_redraw_seeded(k)`, replay mode |
| `runcoms.c` | `src/frontend/` | `mc_params_only` loop and `monte_seed` single-run orchestration |
| `outitf.c` | `src/frontend/` | `ft_norefprint` flag gating reference-value prints |
| `options.c` | `src/frontend/` | `norefvalue` → `ft_norefprint` plumbing |
| `spiceif.c` | `src/frontend/` | Mirror of `norefvalue` plumbing |
