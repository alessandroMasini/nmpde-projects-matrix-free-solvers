"""@file plots.py
@brief This file contains all the necessary functions for plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Columns and Directory Parameters
# -----------------------------------------------------------------------------

FILE_COLUMNS = ["max_iter", "tol", "t_step", "it_n", "err", "total_t", "l2_error"]
DIR_PARAMS = ["solver", "problem", "n_additional_refinements", "n_procs", "n_threads", "simd"]

X_PARAMS = ["n_additional_refinements", "n_procs", "n_threads", "simd", "tol", "t_step"]
Y_PARAMS = ["it_n", "total_t", "l2_error"]


# Algo directory names are too expressive, and they need to be remapped in order to be actually usable
# algo_name = {"cmaes_mpi_1": "cmaes", "cmaes_mpi_2": "cmaes", "cmaes_serial": "cmaes", "pso_serial": "pso", "pso_mpi": "pso", "rcga_serial": "rcga", "rcga_mpi": "rcga", "de_serial": "de", "de_mpi": "de"}

def comparison_name(key, value):
    return str(value)

def load_file_data(file_path):
    with open(file_path, "r") as f:
        header = f.readline().strip().split()
    data = np.loadtxt(file_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data

def extract_dir_params(path_parts):
    """Extract directory parameters from path parts."""
    return {
        "solver": path_parts[-6],
        "problem": path_parts[-5],
        "n_additional_refinements": int(path_parts[-4]),
        "n_procs": int(path_parts[-3]),
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
        if key in FILE_COLUMNS:
            idx = header.index(key)
            if not np.allclose(data[:, idx], val):
                return False
        elif key in DIR_PARAMS:
            continue
        else:
            raise ValueError(f"Unknown fixed parameter: {key}")
    
    # Checking that the value is between the allowed for comparison
    if compare_values_info[0] and not compare_values_info[1] and str(compare) in FILE_COLUMNS:
        idx = header.index(str(compare))
        flag = False
        for el in compare_values:
            if np.allclose(data[:, idx], float(el)):
                flag = True
        
        return flag


    return True

def actually_finished(file_data):
    header, data = file_data
    tol_idx = header.index("tol")
    err_idx = header.index("err")

    converged = float(data[-1][tol_idx]) >= float(data[-1][err_idx])

    return converged


def extract_data(dir_params, file_data, x, all):
    header, data = file_data

    if all:
        if x in DIR_PARAMS:
            return np.array([float(dir_params[x])] * len(data))

        if x in FILE_COLUMNS:
            return data[:, header.index(x)]
    else: 
        if x in DIR_PARAMS:
            return np.array([float(dir_params[x])])

        if x in FILE_COLUMNS:
            return np.array([data[-1, header.index(x)]])

    raise ValueError(f"Unknown parameter: {x}")


def stop_searching(fixed_params, key, dir, compare, compare_values_flags, compare_values):
    if not dir.is_dir():
        return True

    elif key in fixed_params and str(fixed_params[key]) != comparison_name(key, dir.name):
        return True
        
    if compare_values_flags[0] and not compare_values_flags[1] and str(compare) == key and comparison_name(key, dir.name) not in compare_values[1]:
        return True
    
    return False

# -----------------------------------------------------------------------------
# Aggregated Plots
# -----------------------------------------------------------------------------

def plot_average_results(fixed_params, x, y, req_finished, compare, compare_values, save_path, scalex, scaley, plot_theor):
    tests_dir = Path("./tests")
    required_finish = bool(req_finished)
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

    # Tree traversal.
    # A directory is skipped if it is not in the fixed parameters
    # or among the values chosen for comparison
    for solver_dir in tests_dir.iterdir():          
        if stop_searching(fixed_params, "solver", solver_dir, compare, compare_values_info, compare_values):
            continue

        for problem_dir in solver_dir.iterdir():
            if stop_searching(fixed_params, "problem", problem_dir, compare, compare_values_info, compare_values):
                continue

            for n_additional_refinements_dir in problem_dir.iterdir():
                if stop_searching(fixed_params, "n_additional_refinements", n_additional_refinements_dir, compare, compare_values_info, compare_values):
                    continue

                for n_procs_dir in n_additional_refinements_dir.iterdir():
                    if stop_searching(fixed_params, "n_procs", n_procs_dir, compare, compare_values_info, compare_values):
                        continue
                
                    for n_threads_dir in n_procs_dir.iterdir():
                        if stop_searching(fixed_params, "n_threads", n_threads_dir, compare, compare_values_info, compare_values):
                            continue

                        for simd_dir in n_threads_dir.iterdir():
                            if stop_searching(fixed_params, "simd", simd_dir, compare, compare_values_info, compare_values):
                                continue
                                
                            # At this point, all test parameters from the
                            # directory can be extracted and saved
                            dir_params = extract_dir_params(simd_dir.parts)

                            # Looping among the various identical problems
                            # Note: solver settings may still change, i.e.
                            #       we may still have different tol, 
                            #       max_iter values
                            for test in simd_dir.iterdir():
                                file_data = load_file_data(str(test) + "/log.txt")

                                # A file is skipped if it is does not respect the fixed parameters
                                # or its values are not among the ones chosen for comparison
                                if not satisfies_fixed_params(file_data, fixed_params, compare, compare_values_info, compare_values):
                                    continue
                                
                                # At this point, only if the test did not converge the
                                # run is discarded
                                potential += 1

                                if required_finish and not actually_finished(file_data):
                                    discarded += 1
                                    continue

                                # If the current value for the comparing attribute has not yet been seen, we need to update the list of possible values for the comparison
                                if compare_values_info[0] and compare_values_info[1]:
                                    if comparison_name(compare, dir_params[compare]) not in curves.keys():
                                        curves[comparison_name(compare, dir_params[compare])] = []
                                
                                # Only if the number of iterations is considered on the x axis
                                # it makes sense to take all points. Otherwise, only the last needs to be considered.
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
    if not curves:
        print(f"[WARNING] No matching runs found for {fixed_params}. Empty plot skipped.")
        return 1
    
    # Utility variables for theoretical scaling lines
    p = []
    base_time = {}
    
    plt.figure(figsize=(10, 6))

    if compare is not None:
        for compare_value, value_list in curves.items():
            bucket = defaultdict(list)
            for x_arr, y_arr in value_list:
                for xi, yi in zip(x_arr, y_arr):
                    bucket[xi].append(yi)

            xs = np.array(sorted(bucket.keys()))

            means = np.array([np.mean(bucket[xv]) for xv in xs])
            if x in ["n_threads", "n_procs"]:
                base_time[compare_value] = means[0]
            stds  = np.array([np.std(bucket[xv]) for xv in xs])

            upper = means + 2 * stds
            lower = means - 2 * stds

            # The confidence interval of time cannot go below 0
            lower = np.maximum(lower, np.zeros_like(lower))
            
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
        if x in ["n_threads", "n_procs"]:
            base_time = means[0]
        stds  = np.array([np.std(bucket[xv]) for xv in xs])

        upper = means + 2 * stds
        lower = means - 2 * stds

        # The confidence interval of time cannot go below 0
        lower = np.maximum(lower, np.zeros_like(lower))
        
        p = plt.plot(xs, means)
        plt.fill_between(xs, lower, upper, alpha=0.3)

    
    # Inserting optimal scaling line, if necessary
    if plot_theor:
        # Since the theoretical plotting line is considered by taking
        # serial_time / (n_procs * n_cores)
        # we need to retrieve the other divisor that is not the currently
        # considered x axis point.
        # For example, if on the x axis we have n_procs, we need to retrieve
        # the test's n_cores.
        divisor = 0
        if x == "n_threads":
            divisor = dir_params["n_cores"]
        elif x == "n_cores":
            divisor = dir_params["n_threads"]

        if compare is not None:
            i = 0
            for compare_value, value_list in curves.items():
                bucket_theoretical = defaultdict(list)

                # We need to consider only the array of x_points, since the
                # times are derived by dividing the base_time recorded
                # in the first loop. They are stored in curves[0]
                for xi in curves[0]:
                    bucket_theoretical[xi].append(base_time[compare_value]/(xi * divisor))

                xs = np.array(sorted(bucket_theoretical.keys()))
                means = np.array([np.mean(bucket_theoretical[xv]) for xv in xs])

                plt.plot(xs, means, "--", color = p[i][0].get_color())
                i+=1

        else: 
            bucket_theoretical = defaultdict(list)

            # We need to consider only the array of x_points, since the
            # times are derived by dividing the base_time recorded
            # in the first loop. They are stored in curves[0]
            for xi in curves[0]:
                bucket_theoretical[xi].append(base_time/(xi * divisor))

            xs = np.array(sorted(bucket_theoretical.keys()))
            means = np.array([np.mean(bucket_theoretical[xv]) for xv in xs])

            plt.plot(xs, means, "--", color = p[0].get_color())

    xlegend = int(logxscale) * " (log scale)"
    ylegend = int(logyscale) * " (log scale)"
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
    # Actually finished
    # Allowed: 0, 1
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--finished",
        type=int,
        help="1: only runs that converged will be considered. 0: all of them",
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

    # -------------------------------------------------------------------------
    # Build fixed_params dictionary
    # Only include parameters the user actually provided
    # -------------------------------------------------------------------------
    fixed_params = {}
    output_name = args.x + "_vs_" + args.y + "___"
    for p in DIR_PARAMS + FILE_COLUMNS:
        if p in ["err", "total_t"]:
            continue

        val = getattr(args, p, None)
        if val is not None:
            # Update output name
            output_name += p + "-" + val + "---" 

            # Convert integer-valued directory params properly
            if p in ["n_additional_refinements", "n_procs", "n_threads", "simd, it_n"]:
                val = int(val)
            elif p in ["tol"]:
                val = float(val)
            fixed_params[p] = val

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

    # Correctness checking for theoretical scaling
    if bool(getattr(args, "scale_line")) and args.x not in ["n_procs", "n_threads"]:
        print("Error, you can show a scaling plot only if you are plotting against the number of processes or threads")
        no_error = False

    if getattr(args, "finished") is not None and args.finished == 1:
        output_name += "finished---"

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
        discarded = plot_average_results(fixed_params, args.x, args.y, args.finished, args.compare, args.compare_only, output_name, logxscale, logyscale, args.scale_line)
    
        print(f"A fraction of {discarded} tests did not actually converge and were not plotted")