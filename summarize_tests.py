#!/usr/bin/env python3
"""
Summarize extensive test results into a formatted table.

Table columns:
  solver problem n_procs n_threads simd n_additional_refinements delta_t
  it_n err converged total_t %err

Identical runs (same first 7 params) are aggregated with averages.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import statistics


FILE_COLUMNS = ["delta_t", "max_iter", "tol", "rel_t_step", "it_n", "err", "total_t", "l2_error", "h1_error", "linfty_error", "converged"]

# The summary script reads the current log contract. Every row is expected to
# contain exactly these whitespace-separated columns, in this order.

@dataclass
class TestResult:
    solver: str
    problem: str
    n_procs: int
    n_threads: int
    simd: int
    n_additional_refinements: int
    delta_t: float

    it_n: Optional[float] = None
    err: Optional[float] = None
    converged: Optional[float] = None
    total_t: Optional[float] = None
    has_log: bool = False

    def key(self) -> Tuple:
        """Return key for grouping identical runs."""
        return (self.solver, self.problem, self.n_procs, self.n_threads,
                self.simd, self.n_additional_refinements, self.delta_t)


def normalize_column_name(column: str) -> str:
    """Map a raw log token to the column name used by the summarizer."""
    # Strip deallog's prefix before interpreting the physical meaning of the
    # column. The prefix is about where the line came from, not about the data.
    return column.removeprefix("DEAL::")


def normalize_header(header_line: str) -> List[str]:
    """Recognize the meaning of each log column in the current layout."""
    raw_header = header_line.strip().split()

    # deallog can emit DEAL:: as its own token. Remove it so the comparison is
    # made only against solver quantities. This must happen before name
    # normalization, because stripping DEAL:: from a standalone prefix would
    # otherwise leave an empty pseudo-column.
    if raw_header and raw_header[0] == "DEAL::":
        raw_header = raw_header[1:]

    header = [normalize_column_name(column) for column in raw_header]

    # Be strict about the file format. A summary table with shifted columns is
    # worse than no table, because it looks authoritative while being wrong.
    if header != FILE_COLUMNS:
        raise ValueError(
            "Unexpected log header. Expected "
            + " ".join(FILE_COLUMNS)
            + ", got "
            + " ".join(header)
        )

    return header


def parse_data_line(line: str, n_columns: int) -> Optional[List[str]]:
    """Parse one row from the current whitespace-separated log layout."""
    parts = line.split()

    # Handle the same optional deallog prefix as the header path, before any
    # interpretation of the row as numerical data.
    if parts and parts[0] == "DEAL::":
        parts = parts[1:]

    if len(parts) != n_columns:
        raise ValueError(
            f"Unexpected number of log columns: expected {n_columns}, got {len(parts)}"
        )

    return parts


def parse_log_file(log_path: Path) -> Optional[Dict]:
    """Parse a log.txt file and extract statistics.

    For time-dependent problems, each time step may have multiple iterations.
    We track statistics per time step, then average across time steps. The
    convergence result is kept as the solver-level flag written by SolverControl
    whenever that flag is available, because it is the closest representation of
    what the linear solver actually decided.
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except OSError:
        return None

    if len(lines) < 2:
        return None

    # Validate the header once. After this point every dictionary lookup below
    # is about the experiment, not about guessing the file shape.
    header = normalize_header(lines[0])

    # Blank lines do not carry solver information, so they should not create
    # empty pseudo-time-steps in the aggregate.
    data_lines = [line for line in lines[1:] if line.strip()]
    if not data_lines:
        return None

    # Time-dependent runs print one block per relative time step. Grouping rows
    # this way lets the table report a representative iteration count and
    # residual for the whole run instead of over-weighting problems with more
    # internal solver iterations.
    time_steps = {}  # rel_t_step -> list of (it_n, err)
    delta_t = None
    total_t = None
    overall_converged = None

    for line in data_lines:
        # A malformed row should stop the summary instead of being interpreted
        # as "this test has no log". The current C++ logger writes a fixed set
        # of whitespace-separated columns, so anything else is a real mismatch.
        parts = parse_data_line(line, len(header))
        row = dict(zip(header, parts))

        delta_t = float(row['delta_t'])
        rel_t_step = float(row['rel_t_step'])
        it_n = int(row['it_n'])
        err = float(row['err'])
        total_t = float(row['total_t'])

        # This flag is written by SolverControl and answers the run status
        # directly. Do not infer convergence from residuals here: a failed solve
        # can still leave a small-looking final number in some edge cases, and
        # the solver already made the decision.
        overall_converged = int(float(row['converged']))

        if rel_t_step not in time_steps:
            time_steps[rel_t_step] = []
        time_steps[rel_t_step].append((it_n, err))

    if not time_steps:
        return None

    # For each representative time position, the last row is the final residual
    # checkpoint for that slice. Averaging those endpoints gives one compact
    # result per test directory.
    time_step_it_ns = []
    time_step_errs = []

    for ts_data in time_steps.values():
        final_it_n, final_err = ts_data[-1]
        time_step_it_ns.append(final_it_n)
        time_step_errs.append(final_err)

    return {
        'it_n': statistics.mean(time_step_it_ns),
        'err': statistics.mean(time_step_errs),
        'converged': overall_converged,
        'total_t': total_t,
        'delta_t': delta_t,
    }


def collect_test_results(tests_dir: Path) -> List[TestResult]:
    """Collect all test results from directory structure."""
    results = []

    # Directory structure: tests/{solver}/{problem}/{n_add_ref}/{n_procs}/{n_threads}/{simd}/test_N/log.txt
    for solver_dir in tests_dir.iterdir():
        if not solver_dir.is_dir():
            continue
        solver = solver_dir.name

        for problem_dir in solver_dir.iterdir():
            if not problem_dir.is_dir():
                continue
            problem = problem_dir.name

            for n_add_ref_dir in problem_dir.iterdir():
                if not n_add_ref_dir.is_dir():
                    continue
                try:
                    n_additional_refinements = int(n_add_ref_dir.name)
                except ValueError:
                    continue

                for n_procs_dir in n_add_ref_dir.iterdir():
                    if not n_procs_dir.is_dir():
                        continue
                    try:
                        n_procs = int(n_procs_dir.name)
                    except ValueError:
                        continue

                    for n_threads_dir in n_procs_dir.iterdir():
                        if not n_threads_dir.is_dir():
                            continue
                        try:
                            n_threads = int(n_threads_dir.name)
                        except ValueError:
                            continue

                        for simd_dir in n_threads_dir.iterdir():
                            if not simd_dir.is_dir():
                                continue
                            try:
                                simd = int(simd_dir.name)
                            except ValueError:
                                continue

                            # Iterate over test_N directories
                            for test_dir in simd_dir.iterdir():
                                if not test_dir.is_dir() or not test_dir.name.startswith('test_'):
                                    continue

                                log_file = test_dir / 'log.txt'
                                result = TestResult(
                                    solver=solver,
                                    problem=problem,
                                    n_procs=n_procs,
                                    n_threads=n_threads,
                                    simd=simd,
                                    n_additional_refinements=n_additional_refinements,
                                    delta_t=0.0,  # Will be set from log file
                                )

                                if log_file.exists():
                                    parsed = parse_log_file(log_file)
                                    if parsed:
                                        result.delta_t = parsed['delta_t']
                                        result.it_n = parsed['it_n']
                                        result.err = parsed['err']
                                        result.converged = parsed['converged']
                                        result.total_t = parsed['total_t']
                                        result.has_log = True

                                results.append(result)

    return results


def aggregate_results(results: List[TestResult]) -> Dict[Tuple, Dict]:
    """Group identical runs and calculate averages."""
    grouped = defaultdict(list)

    for result in results:
        grouped[result.key()].append(result)

    aggregated = {}
    for key, group in grouped.items():
        # Count how many have log files
        with_log = sum(1 for r in group if r.has_log)
        total = len(group)
        missing_pct = 100 * (total - with_log) / total if total > 0 else 0

        # Average statistics from runs with logs
        runs_with_logs = [r for r in group if r.has_log]

        if runs_with_logs:
            avg_it_n = statistics.mean(r.it_n for r in runs_with_logs)
            avg_err = statistics.mean(r.err for r in runs_with_logs)
            avg_converged = statistics.mean(r.converged for r in runs_with_logs)
            avg_total_t = statistics.mean(r.total_t for r in runs_with_logs)
        else:
            avg_it_n = avg_err = avg_converged = avg_total_t = 0

        aggregated[key] = {
            'it_n': avg_it_n,
            'err': avg_err,
            'converged': avg_converged,
            'total_t': avg_total_t,
            'missing_pct': missing_pct,
            'total_runs': total,
        }

    return aggregated


def format_table(aggregated: Dict[Tuple, Dict]) -> str:
    """Format aggregated results as a pretty table."""
    if not aggregated:
        return "No results to display."

    lines = []
    col_widths = {
        'solver': 14,
        'problem': 14,
        'n_procs': 10,
        'n_threads': 11,
        'simd': 7,
        'n_add_ref': 12,
        'delta_t': 12,
        'it_n': 12,
        'err': 15,
        'converged': 10,
        'total_t': 12,
        'err_pct': 8,
    }

    # Header with column names
    header = (
        f"{'Solver':<{col_widths['solver']}} "
        f"{'Problem':<{col_widths['problem']}} "
        f"{'n_procs':>{col_widths['n_procs']}} "
        f"{'n_threads':>{col_widths['n_threads']}} "
        f"{'simd':>{col_widths['simd']}} "
        f"{'n_add_ref':>{col_widths['n_add_ref']}} "
        f"{'delta_t':>{col_widths['delta_t']}} "
        f"{'it_n':>{col_widths['it_n']}} "
        f"{'err':>{col_widths['err']}} "
        f"{'conv':>{col_widths['converged']}} "
        f"{'total_t':>{col_widths['total_t']}} "
        f"{'%err':>{col_widths['err_pct']}}"
    )
    separator = "─" * len(header)

    lines.append(header)
    lines.append(separator)

    # Sort by key for consistent output
    for key in sorted(aggregated.keys()):
        solver, problem, n_procs, n_threads, simd, n_add_ref, delta_t = key
        stats = aggregated[key]

        line = (
            f"{solver:<{col_widths['solver']}} "
            f"{problem:<{col_widths['problem']}} "
            f"{n_procs:>{col_widths['n_procs']}} "
            f"{n_threads:>{col_widths['n_threads']}} "
            f"{simd:>{col_widths['simd']}} "
            f"{n_add_ref:>{col_widths['n_add_ref']}} "
            f"{delta_t:>{col_widths['delta_t']}.6f} "
            f"{stats['it_n']:>{col_widths['it_n']}.2f} "
            f"{stats['err']:>{col_widths['err']}.6e} "
            f"{stats['converged']:>{col_widths['converged']}.2f} "
            f"{stats['total_t']:>{col_widths['total_t']}.4f} "
            f"{stats['missing_pct']:>{col_widths['err_pct']-1}.1f}%"
        )
        lines.append(line)

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        tests_dir = Path(sys.argv[1])
    else:
        tests_dir = Path(__file__).parent / "tests"

    if not tests_dir.exists():
        print(f"Error: tests directory not found at {tests_dir}")
        sys.exit(1)

    print("Collecting test results...")
    results = collect_test_results(tests_dir)

    if not results:
        print("No test results found!")
        sys.exit(1)

    print(f"Found {len(results)} test runs")

    print("Aggregating results...")
    aggregated = aggregate_results(results)

    print("\n" + "=" * 120)
    print("TEST SUMMARY")
    print("=" * 120)
    print(format_table(aggregated))
    print("=" * 120)

    return 0


if __name__ == "__main__":
    sys.exit(main())
