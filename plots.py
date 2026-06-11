"""@file plots.py
@brief This file contains all the necessary functions for plotting.
"""

import os
import getpass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Columns and Directory Parameters
# -----------------------------------------------------------------------------

FILE_COLUMNS = ["delta_t", "max_iter", "adj_tol", "rel_t_step", "it_n", "err", "total_t", "l2_error", "h1_error", "linfty_error", "converged"]
LEGACY_FILE_COLUMNS = ["delta_t", "max_iter", "tol", "rel_t_step", "it_n", "err", "total_t", "l2_error", "h1_error", "linfty_error", "converged"]
DIR_PARAMS = ["solver", "problem", "fe_deg", "n_additional_refinements", "n_ranks", "n_threads", "simd"]

X_PARAMS = ["delta_t", "fe_deg", "n_additional_refinements", "n_ranks", "n_threads", "simd", "adj_tol", "rel_t_step", "it_n"]
Y_PARAMS = ["it_n", "total_t", "l2_error", "h1_error", "linfty_error", "mf_speedup"]
MF_SPEEDUP_SOLVERS = ("matrix_based", "matrix_free")
SETTING_FILE_COLUMNS = ["delta_t", "max_iter", "rel_t_step"]

# Keep the plotting-side problem validation aligned with the solver CLIs.
SUPPORTED_PROBLEMS = {"advanced", "lab_02", "lab_03", "parabolic", "transient", "mms"}

def problem_dimension(problem):
    """Return the dimension associated with a solver problem name."""
    # The C++ drivers instantiate lab_02 in 2D and all other supported CLI
    # problems in 3D, so the theoretical refinement line must follow the same
    # mapping instead of guessing from the plotted data.
    if problem not in SUPPORTED_PROBLEMS:
        raise ValueError(
            f"unknown problem '{problem}'. Expected one of {sorted(SUPPORTED_PROBLEMS)}"
        )

    return 2 if problem == "lab_02" else 3

def scale_line_mode(x, y, compare):
    """Classify which theoretical scaling line, if any, the plot can show."""
    # Refinement-growth scaling is defined only for final run time plotted
    # against the number of additional refinements.
    if x == "n_additional_refinements" and y == "total_t":
        return "refinement_time"

    # Existing strong-scaling behavior is kept for plots where MPI ranks are
    # either the x axis or the comparison variable.
    if x == "n_ranks" or compare == "n_ranks":
        return "strong_scaling"

    return None

def theoretical_refinement_times(xs, means, dim):
    """Build the ideal time-growth line for refinement sweeps."""
    # The measured leftmost point anchors the line, exactly as requested by
    # t'(0) = t(0).
    base_time = means[0]

    # Each further point uses the plotted refinement value as ref(i), giving
    # t'(i) = 2^(dim * ref(i)) * t(0).
    theoretical = np.array([
        (2 ** (dim * int(refinement))) * base_time
        for refinement in xs
    ])

    # This explicit overwrite preserves the anchor even when the first shown
    # refinement is not zero.
    theoretical[0] = base_time
    return theoretical

def legacy_fe_degree_for_solver(solver):
    """Return the old hard-coded FE degree for pre-fe_deg result folders."""
    # New output paths carry fe_deg explicitly. The fallback lets older result
    # trees remain plottable without losing the fact that old matrix-free runs
    # used degree 4 while old matrix-based runs used degree 2.
    return 4 if solver == "matrix_free" else 2

def scratch_global_tests_dir(root=None):
    '''A highly redundant function to retrieve files in case they are saved to 
    scratch global.'''
    if root is not None:
        return root / "tests"

    direct = os.environ.get("MFSOLVER_SCRATCH_GLOBAL_TESTS_DIR")
    if direct:
        return Path(direct)

    root_from_env = os.environ.get("MFSOLVER_SCRATCH_GLOBAL_ROOT")
    if root_from_env:
        return Path(root_from_env) / "tests"

    user_name = os.environ.get("USER") or getpass.getuser()
    repo_name = Path(__file__).resolve().parent.name
    return Path("/scratch_global") / user_name / repo_name / "tests"


def resolve_tests_dir(tests_dir=None, from_scratch_global=False, scratch_global_root=None):
    if from_scratch_global or scratch_global_root is not None:
        if tests_dir is not None:
            raise ValueError("provide either --tests_dir or --from-scratch-global, not both")
        return scratch_global_tests_dir(scratch_global_root)

    return tests_dir or Path("./tests")

# Log files are the interface between the C++ solvers and the plotting code.
# FILE_COLUMNS is the current contract: every log row must contain exactly
# these whitespace-separated quantities, in this order.

def comparison_name(key, value):
    return str(value)

def comparison_sort_key(value):
    """Sort numeric comparison values numerically and text values alphabetically."""
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))

def fixed_param_value(key, value):
    """Convert command-line fixed parameters to the type used internally."""
    if key in ["fe_deg", "n_additional_refinements", "n_ranks", "n_threads", "simd", "it_n", "max_iter", "converged"]:
        return int(value)

    if key in ["delta_t", "adj_tol", "rel_t_step", "l2_error", "h1_error", "linfty_error"]:
        return float(value)

    return value

def normalize_column_name(column):
    """Return the semantic column name used by the plotting code."""
    # deallog may prefix the first token with DEAL::. That prefix describes the
    # logging channel, not the physical quantity in the column.
    return column.removeprefix("DEAL::")

def column_index(header, column):
    return header.index(normalize_column_name(column))

def is_file_column(column):
    return column in FILE_COLUMNS or column in LEGACY_FILE_COLUMNS

def normalize_header(header_line):
    """Interpret a log header according to the current solver-to-script schema."""
    raw_header = header_line.strip().split()

    # A standalone DEAL:: token can appear when deallog separates its prefix
    # from the first real column. Remove it before normalizing names; otherwise
    # "DEAL::".removeprefix("DEAL::") becomes an empty pseudo-column.
    if raw_header and raw_header[0] == "DEAL::":
        raw_header = raw_header[1:]

    header = [normalize_column_name(column) for column in raw_header]

    # From here on, be strict. If the C++ output changes, the scripts should
    # fail loudly instead of silently plotting numbers under the wrong labels.
    if header == FILE_COLUMNS or header == LEGACY_FILE_COLUMNS:
        return header

    else:
        raise ValueError(
            "Unexpected log header. Expected "
            + " ".join(FILE_COLUMNS)
            + " or "
            + " ".join(LEGACY_FILE_COLUMNS)
            + ", got "
            + " ".join(header)
        )

def parse_data_line(line, n_columns):
    """Parse one row from the current whitespace-separated log layout."""
    parts = line.split()

    # Match the same optional deallog prefix handled for the header. After this,
    # every remaining token should be one numeric column. The check happens
    # before normalize_column_name for the same reason as in normalize_header:
    # DEAL:: alone is a log prefix, not an empty data field.
    if parts and parts[0] == "DEAL::":
        parts = parts[1:]

    if len(parts) != n_columns:
        raise ValueError(
            f"Unexpected number of log columns: expected {n_columns}, got {len(parts)}"
        )

    return parts

def load_file_data(file_path):
    """Load a log file as numeric data while preserving column meaning."""
    with open(file_path, "r") as f:
        # The header tells the plotting code what each numeric column means.
        # We validate it once so the rest of the code can use column names
        # without repeatedly checking the file layout.
        header = normalize_header(f.readline())

        # Empty lines are harmless, but every non-empty line must be a complete
        # row in the current log format.
        rows = [
            parse_data_line(line, len(header))
            for line in f
            if line.strip()
        ]

    # Convert only after parsing all rows so a malformed row produces a useful
    # format error rather than a confusing partial NumPy array.
    data = np.array(rows, dtype=float)

    # np.array collapses a single-row file to one dimension; the plotting code
    # always works with "rows x columns", so put that dimension back.
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data

def extract_dir_params(path_parts):
    """Extract directory parameters from path parts."""
    # New layout:
    # tests/{solver}/{problem}/{fe_deg}/{n_add_ref}/{n_ranks}/{n_threads}/{simd}/test_N
    try:
        return {
            "solver": path_parts[-7],
            "problem": path_parts[-6],
            "fe_deg": int(path_parts[-5]),
            "n_additional_refinements": int(path_parts[-4]),
            "n_ranks": int(path_parts[-3]),
            "n_threads": int(path_parts[-2]),
            "simd": int(path_parts[-1]),
        }
    except (IndexError, ValueError):
        pass

    # Legacy layout before fe_deg was a directory level:
    # tests/{solver}/{problem}/{n_add_ref}/{n_ranks}/{n_threads}/{simd}/test_N
    solver = path_parts[-6]
    return {
        "solver": solver,
        "problem": path_parts[-5],
        "fe_deg": legacy_fe_degree_for_solver(solver),
        "n_additional_refinements": int(path_parts[-4]),
        "n_ranks": int(path_parts[-3]),
        "n_threads": int(path_parts[-2]),
        "simd": int(path_parts[-1]),
    }


def satisfies_fixed_params(file_data, fixed_params, compare, compare_values_info, compare_values):
    header, data = file_data

    # Checking that the value is between the fixed ones
    for key, val in fixed_params.items():
        # -----------------------------
        # Parameters coming from file columns
        # -----------------------------
        if is_file_column(key):
            idx = column_index(header, key)
            if not np.allclose(data[:, idx], val):
                return False
        elif key in DIR_PARAMS:
            continue
        else:
            raise ValueError(f"Unknown fixed parameter: {key}")
    
    # Checking that the value is between the allowed for comparison
    if compare_values_info[0] and not compare_values_info[1] and is_file_column(str(compare)):
        idx = column_index(header, str(compare))
        flag = False
        for el in compare_values:
            if np.allclose(data[:, idx], float(el)):
                flag = True
        
        return flag


    return True

def actually_converged(file_data):
    """Decide whether a run converged using the solver's own convergence flag.

    The final residual is still useful for plotting, but it is not the
    authoritative answer. SolverControl already decided whether the solve
    converged, and the current log format writes that decision explicitly.
    """
    header, data = file_data

    # The last row is enough because the convergence flag is repeated on every
    # residual-history row by the C++ logger.
    converged_idx = column_index(header, "converged")
    return bool(int(data[-1][converged_idx]))


def extract_data(dir_params, file_data, x, all):
    header, data = file_data

    if all:
        if x in DIR_PARAMS:
            return np.array([float(dir_params[x])] * len(data))

        if is_file_column(x):
            return data[:, column_index(header, x)]
    else: 
        if x in DIR_PARAMS:
            return np.array([float(dir_params[x])])

        if is_file_column(x):
            return np.array([data[-1, column_index(header, x)]])

    raise ValueError(f"Unknown parameter: {x}")

def speedup_setting_key(dir_params, file_data, simd_override=None):
    """Build the solver-independent key used to pair MB and MF timings."""
    header, data = file_data
    simd_value = dir_params["simd"] if simd_override is None else simd_override

    # Solver is deliberately excluded: the key describes the numerical/test
    # setting that must be identical before the two solver timings can be
    # divided.
    key_parts = [
        ("problem", dir_params["problem"]),
        ("fe_deg", dir_params["fe_deg"]),
        ("n_additional_refinements", dir_params["n_additional_refinements"]),
        ("n_ranks", dir_params["n_ranks"]),
        ("n_threads", dir_params["n_threads"]),
        ("simd", simd_value),
    ]

    # These columns are run inputs written in the log. Output quantities such
    # as residuals and errors are not part of the pairing key because they may
    # legitimately differ between matrix-based and matrix-free runs.
    for column in SETTING_FILE_COLUMNS:
        if column in header:
            key_parts.append((column, float(data[-1, column_index(header, column)])))

    return tuple(key_parts)

def setting_description(setting_key):
    """Format a pairing key in warning/error messages."""
    return ", ".join(f"{key}={value}" for key, value in setting_key)

def add_speedup_sample(curves, compare, compare_value, x_value, speedup):
    """Append one derived speedup datapoint to the normal curve container."""
    x_arr = np.array([float(x_value)])
    y_arr = np.array([float(speedup)])

    if compare is not None:
        curves[compare_value].append((x_arr, y_arr))
    else:
        curves.append((x_arr, y_arr))

def curves_have_samples(curves, compare):
    """Return whether the curve container has at least one plottable sample."""
    if compare is not None:
        return any(bool(value_list) for value_list in curves.values())

    return bool(curves)

def collect_mf_speedup_curves(fixed_params, x, req_converged, compare, compare_values, tests_dir):
    """Collect matrix_based/matrix_free runtime ratios as plot samples."""
    tests_dir = Path(tests_dir)
    required_convergence = bool(req_converged)

    if compare is None or compare_values is None:
        compare_values_info = [compare is not None, True]
        curves = defaultdict(list) if compare is not None else []
    else:
        compare_values_info = [True, False]
        curves = {str(el): [] for el in compare_values}

    runs_by_setting = defaultdict(lambda: {
        "times": defaultdict(list),
        "x_values": defaultdict(list),
        "compare_value": None,
    })
    matrix_based_simd0_cache = defaultdict(lambda: {
        "times": [],
        "x_values": [],
    })
    seen_solvers = set()
    potential = 0
    discarded = 0

    # First pass: gather all matching logs by solver-independent setting.
    for log_file in tests_dir.rglob("log.txt"):
        test_dir = log_file.parent
        if not test_dir.name.startswith("test_"):
            continue

        try:
            dir_params = extract_dir_params(test_dir.parent.parts)
        except (IndexError, ValueError):
            continue

        if dir_params["solver"] not in MF_SPEEDUP_SOLVERS:
            continue

        # Matrix-based has only a scalar implementation, so its simd=0 runs are
        # the baseline for every matrix-free SIMD variant. Do not let --simd 1
        # or --compare_only 1 discard that required matrix-based baseline.
        dir_fixed_params = fixed_params
        dir_compare_values_info = compare_values_info
        if dir_params["solver"] == "matrix_based":
            dir_fixed_params = {
                key: value
                for key, value in fixed_params.items()
                if key != "simd"
            }
            dir_compare_values_info = [False, True]

        if not satisfies_dir_params(dir_params, dir_fixed_params, compare, dir_compare_values_info, compare_values):
            continue

        file_data = load_file_data(log_file)

        if not satisfies_fixed_params(file_data, dir_fixed_params, compare, compare_values_info, compare_values):
            continue

        potential += 1

        if required_convergence and not actually_converged(file_data):
            discarded += 1
            continue

        solver = dir_params["solver"]
        seen_solvers.add(solver)
        total_t = extract_data(dir_params, file_data, "total_t", False)[0]

        # Use the selected x value from the same logs. For ordinary setting
        # axes this is identical across solvers; if an output x such as it_n is
        # requested, the matrix_free value is preferred below.
        x_value = extract_data(dir_params, file_data, x, False)[0]

        # Cache matrix-based SIMD-0 and attach it later to every matching
        # matrix-free setting, including matrix-free SIMD-1.
        if solver == "matrix_based":
            if dir_params["simd"] == 0:
                base_key = speedup_setting_key(dir_params, file_data, simd_override=0)
                matrix_based_simd0_cache[base_key]["times"].append(total_t)
                matrix_based_simd0_cache[base_key]["x_values"].append(x_value)
            continue

        setting_key = speedup_setting_key(dir_params, file_data)

        # Store final total_t for this solver and exact setting. Repeated runs
        # are averaged before the ratio is formed.
        runs_by_setting[setting_key]["times"][solver].append(total_t)
        runs_by_setting[setting_key]["x_values"][solver].append(x_value)

        if compare is not None and runs_by_setting[setting_key]["compare_value"] is None:
            runs_by_setting[setting_key]["compare_value"] = comparison_name(compare, dir_params[compare])

    # Fill every matrix-free setting with the matching matrix-based SIMD-0
    # baseline. For matrix-free SIMD-1, the lookup key is normalized to SIMD-0.
    for setting_key, values in runs_by_setting.items():
        if values["times"]["matrix_based"]:
            continue

        base_key = tuple(
            (key, 0 if key == "simd" else value)
            for key, value in setting_key
        )
        if matrix_based_simd0_cache[base_key]["times"]:
            values["times"]["matrix_based"].extend(
                matrix_based_simd0_cache[base_key]["times"]
            )
            values["x_values"]["matrix_based"].extend(
                matrix_based_simd0_cache[base_key]["x_values"]
            )

    # A completely missing solver means the requested speedup is undefined for
    # the whole filtered dataset, so fail instead of producing an empty plot.
    for solver in MF_SPEEDUP_SOLVERS:
        if solver not in seen_solvers:
            print(f"[ERROR] Cannot compute mf_speedup: no matching {solver} data found.")
            return None, potential, discarded, 1

    # Second pass: form ratios for complete settings and warn about holes.
    complete_pairs = 0
    for setting_key, values in sorted(runs_by_setting.items(), key=lambda item: setting_description(item[0])):
        missing_solvers = [
            solver
            for solver in MF_SPEEDUP_SOLVERS
            if not values["times"][solver]
        ]

        if missing_solvers:
            for solver in missing_solvers:
                print(
                    "[WARNING] Missing mf_speedup datapoint: "
                    f"{setting_description(setting_key)}, missing solver={solver}"
                )
            continue

        matrix_based_time = np.mean(values["times"]["matrix_based"])
        matrix_free_time = np.mean(values["times"]["matrix_free"])
        if matrix_free_time == 0:
            print(
                "[WARNING] Missing mf_speedup datapoint: "
                f"{setting_description(setting_key)}, matrix_free total_t is zero"
            )
            continue

        # Prefer the matrix-free x coordinate for output-derived x axes. For
        # setting-derived axes both solvers should provide the same value.
        x_values = values["x_values"]
        x_value = np.mean(x_values["matrix_free"] or x_values["matrix_based"])
        speedup = matrix_based_time / matrix_free_time
        complete_pairs += 1

        if compare is not None:
            compare_value = values["compare_value"]
            if compare_values_info[1] and compare_value not in curves:
                curves[compare_value] = []
            add_speedup_sample(curves, compare, compare_value, x_value, speedup)
        else:
            add_speedup_sample(curves, compare, None, x_value, speedup)

    # If every setting was missing one side of the pair, there is no meaningful
    # speedup plot to render even though both solvers appeared somewhere.
    if complete_pairs == 0:
        print("[ERROR] Cannot compute mf_speedup: no setting has both matrix_based and matrix_free data.")
        return curves, potential, discarded, 1

    return curves, potential, discarded, 0


def stop_searching(fixed_params, key, dir, compare, compare_values_flags, compare_values):
    if not dir.is_dir():
        return True

    elif key in fixed_params and str(fixed_params[key]) != comparison_name(key, dir.name):
        return True
        
    if compare_values_flags[0] and not compare_values_flags[1] and str(compare) == key and comparison_name(key, dir.name) not in compare_values:
        return True
    
    return False


def satisfies_dir_params(dir_params, fixed_params, compare, compare_values_info, compare_values):
    """Check fixed and comparison filters that are encoded in directory names."""
    # Directory parameters are already typed by extract_dir_params(). Filter
    # them before loading log.txt so missing or malformed logs do not matter for
    # configurations the user explicitly excluded.
    for key, val in fixed_params.items():
        if key in DIR_PARAMS and dir_params[key] != val:
            return False

    if (compare_values_info[0] and not compare_values_info[1]
            and str(compare) in DIR_PARAMS):
        return comparison_name(compare, dir_params[compare]) in compare_values

    return True

# -----------------------------------------------------------------------------
# Aggregated Plots
# -----------------------------------------------------------------------------

def plot_average_results(fixed_params, x, y, req_converged, compare, compare_values, save_path, scalex, scaley, plot_theor, tests_dir, scaling_dim=None):
    tests_dir = Path(tests_dir)
    if not tests_dir.exists():
        print(f"[ERROR] tests directory not found at {tests_dir}")
        return 1

    if y == "mf_speedup":
        curves, potential, discarded, status = collect_mf_speedup_curves(
            fixed_params, x, req_converged, compare, compare_values, tests_dir
        )
        if status:
            return None
    else:
        required_convergence = bool(req_converged)
        curves = []
        compare_values_info = []

        # Tracking, if necessary, all the compare values that will be compared upon:
        # The first argument of compare_values_info states
        # whether we need to track comparing values 
        # The second argument states whether we need to track all of them or only some
        if compare is None or compare_values is None:
            compare_values_info = [compare is not None, True]
            if compare is not None:
                curves = defaultdict()
        else:
            compare_values_info = [True, False]
            curves = {str(el): [] for el in compare_values}

        # These variable count the number of runs 
        # that fit the chosen parameters but may be
        # discarded because they did not converge
        potential = 0
        discarded = 0

        # Tree traversal. Recursing from log.txt keeps the plotting code compatible
        # with both the new fe_deg-aware hierarchy and older saved test folders.
        for log_file in tests_dir.rglob("log.txt"):
            test_dir = log_file.parent
            if not test_dir.name.startswith("test_"):
                continue

            try:
                dir_params = extract_dir_params(test_dir.parent.parts)
            except (IndexError, ValueError):
                continue

            if not satisfies_dir_params(dir_params, fixed_params, compare, compare_values_info, compare_values):
                continue

            file_data = load_file_data(log_file)

            # A file is skipped if it does not respect fixed file-column parameters
            # or its values are not among the ones chosen for comparison.
            if not satisfies_fixed_params(file_data, fixed_params, compare, compare_values_info, compare_values):
                continue

            # At this point, only if the test did not converge the run is discarded.
            potential += 1

            if required_convergence and not actually_converged(file_data):
                discarded += 1
                continue

            # If the current value for the comparing attribute has not yet been
            # seen, add a curve bucket for it.
            if compare_values_info[0] and compare_values_info[1]:
                if comparison_name(compare, dir_params[compare]) not in curves.keys():
                    curves[comparison_name(compare, dir_params[compare])] = []

            # Only when the number of iterations is on the x axis does it make
            # sense to take every residual-history point. Other plots use the final
            # row for each run.
            if x == "it_n":
                x_arr = extract_data(dir_params, file_data, x, True)
                y_arr = extract_data(dir_params, file_data, y, True)
            else:
                x_arr = extract_data(dir_params, file_data, x, False)
                y_arr = extract_data(dir_params, file_data, y, False)

            if compare is not None:
                curves[comparison_name(compare, dir_params[compare])].append((x_arr, y_arr))
            else:
                curves.append((x_arr, y_arr))

    frac_discarded = 0

    # If nothing matched, avoid empty plot
    if not curves_have_samples(curves, compare):
        print(f"[WARNING] No matching runs found for {fixed_params}. Empty plot skipped.")
        return 1
    
    # Utility variables for theoretical scaling lines
    p = []
    base_time = {}
    curve_stats = {}
    
    plt.figure(figsize=(10, 6))

    if compare is not None:
        for compare_value, value_list in sorted(curves.items(), key=lambda item: comparison_sort_key(item[0])):
            if not value_list:
                continue

            bucket = defaultdict(list)
            for x_arr, y_arr in value_list:
                for xi, yi in zip(x_arr, y_arr):
                    bucket[xi].append(yi)

            xs = np.array(sorted(bucket.keys()))

            means = np.array([np.mean(bucket[xv]) for xv in xs])

            # If on the x axis we have the number of ranks, the base time is given
            # by the time at the lowest # cores * # cores. This is a number.
            if x == "n_ranks":
                base_time[compare_value] = means[0] * xs[0]
            
            stds  = np.array([np.std(bucket[xv]) for xv in xs])

            upper = means + 2 * stds
            lower = means - 2 * stds

            # The confidence interval of time cannot go below 0
            lower = np.maximum(lower, np.zeros_like(lower))
            curve_stats[compare_value] = (xs, means, stds)
            
            p.append(plt.plot(xs, means, label=compare + " = " + compare_value))
            plt.fill_between(xs, lower, upper, alpha=0.3)

        plt.legend()
    else: 
        bucket = defaultdict(list)
        
        for x_arr, y_arr in curves:
            for xi, yi in zip(x_arr, y_arr):
                bucket[xi].append(yi)

        xs = np.array(sorted(bucket.keys()))

        means = np.array([np.mean(bucket[xv]) for xv in xs])
        if x == "n_ranks":
            base_time = means[0] * xs[0]
        stds  = np.array([np.std(bucket[xv]) for xv in xs])

        upper = means + 2 * stds
        lower = means - 2 * stds

        # The confidence interval of time cannot go below 0
        lower = np.maximum(lower, np.zeros_like(lower))
        
        p = plt.plot(xs, means)
        plt.fill_between(xs, lower, upper, alpha=0.3)

    
    # Inserting optimal scaling line, if necessary
    if plot_theor:
        # Decide once which kind of dashed reference line this plot requires;
        # the CLI has already rejected unsupported combinations.
        theor_mode = scale_line_mode(x, y, compare)

        if compare is not None:
            if theor_mode == "strong_scaling" and compare == "n_ranks":
                min_rank_key = min(curve_stats, key=lambda value: int(value))
                min_rank = int(min_rank_key)
                base_xs, base_means, _ = curve_stats[min_rank_key]

            for i, (compare_value, (xs, means, _)) in enumerate(curve_stats.items()):
                # For refinement sweeps, each compared curve is anchored to
                # its own measured leftmost runtime and then grows like
                # 2^(dim * n_additional_refinements).
                if theor_mode == "refinement_time":
                    theoretical = theoretical_refinement_times(xs, means, scaling_dim)
                    plt.plot(xs, theoretical, "--", color = p[i][0].get_color())

                # When comparing MPI ranks, every ideal line is the measured
                # minimum-rank curve scaled by min_rank / current_rank.
                elif compare == "n_ranks":
                    theoretical = base_means * min_rank / int(compare_value)
                    plt.plot(base_xs, theoretical, "--", color = p[i][0].get_color())
                else:
                    theoretical = base_time[compare_value] / xs
                    plt.plot(xs, theoretical, "--", color = p[i][0].get_color())

        else: 
            xs = np.array(sorted({xi for x_arr, _ in curves for xi in x_arr}))
            means = np.array([np.mean(bucket[xv]) for xv in xs])

            # Refinement sweeps use the problem dimension to model the growth
            # in work after uniform mesh refinement; strong scaling keeps the
            # previous inverse-rank reference.
            if theor_mode == "refinement_time":
                theoretical = theoretical_refinement_times(xs, means, scaling_dim)
            else:
                theoretical = base_time / xs

            plt.plot(xs, theoretical, "--", color = p[0].get_color())

    xlegend = int(scalex) * " (log scale)"
    ylegend = int(scaley) * " (log scale)"
    plt.xlabel(x + xlegend)
    plt.ylabel(y + ylegend)

    title = "Plotting " + y + " vs " + x + " (mean ± 2σ) with settings:\n\n" 
    for key, value in fixed_params.items():
        title += key + ": " + str(value) + "    "
    plt.title(title)
    plt.grid(True)
    if scalex:
        plt.xscale("log")
    if scaley:
        plt.yscale("log")

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/" + save_path + ".png", dpi=300, bbox_inches="tight")
    plt.close()

    if (potential > 0):
        frac_discarded = discarded / potential
    return frac_discarded

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots from experiment outputs.\n All the non-fixed parameter will be averaged upon."
    )

    # -------------------------------------------------------------------------
    # Choice of plot type
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # x and y axis
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--x",
        type=str,
        required=True,
        help=f"x-axis parameter (one of {X_PARAMS})",
    )

    parser.add_argument(
        "--y",
        type=str,
        required=True,
        help=f"y-axis parameter (one of {Y_PARAMS})",
    )

    # -------------------------------------------------------------------------
    # Fixed parameters
    # Allowed: whatever can go as an x parameter
    # -------------------------------------------------------------------------
    for p in X_PARAMS + ["solver", "problem"]:
        parser.add_argument(
            f"--{p}",
            type=str,
            help=f"Fix the parameter '{p}' to a specific value.",
        )

    # -------------------------------------------------------------------------
    # Convergence filter
    # Allowed: 0, 1
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--converged",
        type=int,
        help="1: only converged runs will be considered. 0: converged and overtime runs will be considered",
    )

    # -------------------------------------------------------------------------
    # Use log scale on x axis
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--logxscale",
        type=int,
        help="1: Use log scale on x axis. 0: use linear scale",
    )

     # -------------------------------------------------------------------------
    # Use log scale on y axis
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--logyscale",
        type=int,
        help="1: Use log scale on y axis. 0: use linear scale",
    )

    # -------------------------------------------------------------------------
    # Comparison parameter
    # Allowed: anything that can be on x axis or solver, problem
    # Not allowed: x, y or anything fixed
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--compare",
        type=str,
        help=f"Insert the parameter you want to compare across. All plots will be on the same graph.\nDo not insert the same value of x, y or any fixed parameter. Possible values: {X_PARAMS + ['solver', 'problem']}",
    )

    # -------------------------------------------------------------------------
    # Comparison parameter values
    # Allowed: value related to the compare parameter
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--compare_only",
        type=str,
        help="Insert the values of the compare parameters you want to consider.\nIf absent, all possible values will be taken in consideration.",
    )

    # -------------------------------------------------------------------------
    # Add strong scaling line (allowed for n_threads/n_cores vs final_t)
    # Allowed: 1 for true, 0 for false
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--scale_line",
        type=int,
        help="With 1, it shows what the line showing what strong scaling should look like"
    )

    # -------------------------------------------------------------------------
    # Input tree
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--tests_dir",
        type=Path,
        help="Path to the tests directory.",
    )

    parser.add_argument(
        "--from-scratch-global",
        action="store_true",
        help="Read tests from /scratch_global/$USER/<repo>/tests.",
    )

    parser.add_argument(
        "--scratch-global-root",
        type=Path,
        help="Scratch-global root containing tests/. Implies --from-scratch-global.",
    )

    # -------------------------------------------------------------------------
    # Output file
    # -------------------------------------------------------------------------
    '''
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output file path for the generated plot (.png recommended).",
    )
    '''
    args = parser.parse_args()
    no_error = True

    converged_filter = args.converged
    if converged_filter is not None and converged_filter not in [0, 1]:
        print("Error, --converged must be 0 or 1")
        no_error = False

    if args.x not in X_PARAMS:
        print(f"Error, --x must be one of {X_PARAMS}")
        no_error = False

    if args.y not in Y_PARAMS:
        print(f"Error, --y must be one of {Y_PARAMS}")
        no_error = False

    # Retrieving the test directory
    try:
        tests_dir = resolve_tests_dir(args.tests_dir, args.from_scratch_global, args.scratch_global_root)
    except ValueError as exception:
        print(f"Error: {exception}")
        no_error = False
        tests_dir = Path("./tests")

    # -------------------------------------------------------------------------
    # Build fixed_params dictionary
    # Only include parameters the user actually provided
    # -------------------------------------------------------------------------
    fixed_params = {}
    output_name = args.x + "_vs_" + args.y + "___"
    for p in DIR_PARAMS + FILE_COLUMNS:
        if p in ["err", "total_t", "converged"]:
            continue

        val = getattr(args, p, None)
        if val is not None:
            # Update output name
            output_name += p + "-" + val + "---" 

            fixed_params[p] = fixed_param_value(p, val)

    # Correctness checking for compare variable
    if getattr(args, "compare") is not None:
        if args.compare == args.x or args.compare == args.y:
            print("Error, you cannot compare the variables you are plotting x or y on")
            no_error = False
    
        elif args.compare in fixed_params.keys():
            print("Error, you cannot compare the variables you fixed")
            no_error = False
        
        elif args.compare not in DIR_PARAMS + ["solver", "problem"]:
            print("Error, you cannot compare on this variable")
            no_error = False

        if getattr(args, "compare_only") is not None:
            args.compare_only = args.compare_only.split()

        output_name += "comp_" + args.compare + "---"

    if args.y == "mf_speedup":
        if args.compare == "solver":
            print("Error, --compare solver is disabled when plotting mf_speedup")
            no_error = False

    # Correctness checking for theoretical scaling
    # Theoretical scaling can be shown for strong-scaling rank plots or for
    # total runtime versus additional refinements.
    scaling_dim = None
    if bool(getattr(args, "scale_line")):
        scaling_mode = scale_line_mode(args.x, args.y, args.compare)

        if scaling_mode is None:
            print("Error, --scale_line is available only for rank scaling or for total_t vs n_additional_refinements")
            no_error = False

        # The refinement-time model needs the problem dimension. Require a
        # fixed, valid problem so one plot cannot silently mix 2D and 3D data.
        elif scaling_mode == "refinement_time":
            if "problem" not in fixed_params:
                print("Error, --scale_line with total_t vs n_additional_refinements requires --problem")
                no_error = False
            else:
                try:
                    scaling_dim = problem_dimension(fixed_params["problem"])
                except ValueError as exception:
                    print(f"Error, {exception}")
                    no_error = False

    if converged_filter == 1:
        output_name += "converged---"

    # Trimming useless characters
    if output_name[-3:] in ["---", "___"]:
        output_name = output_name[:-3]

    # Correct axis scale
    logxscale = False
    logyscale = False

    if getattr(args, "logxscale") is not None and args.logxscale == 1:
        logxscale = True

    if getattr(args, "logyscale") is not None and args.logyscale == 1:
        logyscale = True
    
    # -------------------------------------------------------------------------
    # Dispatch plot
    # -------------------------------------------------------------------------
    if no_error:
        discarded = plot_average_results(fixed_params, args.x, args.y, converged_filter, args.compare, args.compare_only, output_name, logxscale, logyscale, args.scale_line, tests_dir, scaling_dim)
    
        if discarded is not None:
            print(f"A fraction of {discarded} tests did not converge and were not plotted")
