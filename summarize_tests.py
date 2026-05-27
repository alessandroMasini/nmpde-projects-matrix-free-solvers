#!/usr/bin/env python3
"""
Summarize extensive test results into a formatted table.

The script treats the on-disk test tree as the source of the experiment
configuration, and the log.txt files as the source of the measured quantities.
Each test_N directory becomes one TestResult. Results with the same
configuration key are then averaged into one table row.

Table columns:
  solver problem n_procs n_threads simd n_add_ref delta_t
  it_n err converged total_t %err

Identical runs (same first 7 params) are aggregated with averages.
"""

import os
import sys
import argparse
import getpass
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import statistics


# The summary script reads the current log contract. Every row is expected to
# contain exactly these whitespace-separated columns, in this order. Some of
# these columns are not printed in the final table, but they are still part of
# the contract because their position keeps the remaining values unambiguous.
FILE_COLUMNS = [
    "delta_t",
    "max_iter",
    "tol",
    "rel_t_step",
    "it_n",
    "err",
    "total_t",
    "l2_error",
    "h1_error",
    "linfty_error",
    "converged",
]

LATEST_RUN_MANIFEST = "latest_run_tests.txt"


def scratch_global_tests_dir(root: Optional[Path] = None) -> Path:
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


def resolve_tests_dir(args: argparse.Namespace) -> Path:
    if args.from_scratch_global or args.scratch_global_root is not None:
        if args.tests_dir is not None:
            raise ValueError("provide either tests_dir or --from-scratch-global, not both")
        return scratch_global_tests_dir(args.scratch_global_root)

    return args.tests_dir or (Path(__file__).parent / "tests")


def manifest_test_dir_candidates(raw_path: str, tests_dir: Path, repo_root: Path) -> List[Path]:
    test_dir = Path(raw_path)
    candidates = []

    if test_dir.is_absolute():
        candidates.append(test_dir)
    else:
        candidates.append(repo_root / test_dir)

    parts = test_dir.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == tests_dir.name and index + 1 < len(parts):
            candidates.append(tests_dir.joinpath(*parts[index + 1:]))
            break

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


@dataclass
class TestResult:
    # One physical test run found under tests/{solver}/{problem}/.../test_N.
    # Configuration fields come from the directory names, while measurement
    # fields come from log.txt when it exists and can be parsed.
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
        """Return key for grouping repeated runs of the same configuration."""
        # test_N is deliberately excluded. Repeated test_N directories under
        # the same configuration are samples of the same experiment.
        return (self.solver, self.problem, self.n_procs, self.n_threads,
                self.simd, self.n_additional_refinements, self.delta_t)


def normalize_column_name(column: str) -> str:
    """Map a raw log token to the column name used by the summarizer."""
    # Keep normalization small and explicit. The only accepted decoration at
    # the moment is deallog's DEAL:: prefix.
    # Strip deallog's prefix before interpreting the physical meaning of the
    # column. The prefix is about where the line came from, not about the data.
    return column.removeprefix("DEAL::")


def normalize_header(header_line: str) -> List[str]:
    """Recognize the meaning of each log column in the current layout."""
    # The header is validated before any row is parsed. That keeps accidental
    # logger changes from silently shifting numeric columns in the summary.
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
    # This mirrors normalize_header(): remove only the optional deallog prefix,
    # then require the remaining data to match the validated column count.
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
    # A missing, unreadable, or empty log is represented as None. The caller
    # keeps the TestResult, but marks it as a run without usable measurements.
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
    # internal solver iterations. Stationary runs naturally form one group,
    # usually at rel_t_step == 0.0.
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

        # The log writer already chose scientific or fixed formatting. Once the
        # columns are known, the summarizer stores only typed values.
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


def make_result_from_test_dir(tests_dir: Path, test_dir: Path) -> Optional[TestResult]:
    """Build one TestResult from a tests/.../test_N directory."""
    try:
        relative_parts = test_dir.relative_to(tests_dir).parts
    except ValueError:
        return None

    if len(relative_parts) != 7:
        return None

    solver, problem, n_add_ref, n_procs, n_threads, simd, test_name = relative_parts

    if not test_name.startswith('test_'):
        return None

    try:
        n_additional_refinements = int(n_add_ref)
        n_procs_value = int(n_procs)
        n_threads_value = int(n_threads)
        simd_value = int(simd)
    except ValueError:
        return None

    result = TestResult(
        solver=solver,
        problem=problem,
        n_procs=n_procs_value,
        n_threads=n_threads_value,
        simd=simd_value,
        n_additional_refinements=n_additional_refinements,
        delta_t=0.0,
    )

    log_file = test_dir / 'log.txt'
    if log_file.exists():
        parsed = parse_log_file(log_file)
        if parsed:
            result.delta_t = parsed['delta_t']
            result.it_n = parsed['it_n']
            result.err = parsed['err']
            result.converged = parsed['converged']
            result.total_t = parsed['total_t']
            result.has_log = True

    return result


def collect_test_results(tests_dir: Path) -> List[TestResult]:
    """Collect all test results from directory structure."""
    # The directory path encodes the parameters that were used to launch the
    # run. Non-directory entries and non-integer parameter folders are ignored
    # so auxiliary files do not break the summary.
    results = []

    # Directory structure:
    # tests/{solver}/{problem}/{n_add_ref}/{n_procs}/{n_threads}/{simd}/test_N/log.txt
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

                            # Each test_N directory is a separate repetition of
                            # the same configuration. Missing logs are kept so
                            # the final %err column can report incomplete runs.
                            for test_dir in simd_dir.iterdir():
                                if not test_dir.is_dir() or not test_dir.name.startswith('test_'):
                                    continue

                                result = make_result_from_test_dir(tests_dir, test_dir)
                                if result is not None:
                                    results.append(result)

    return results


def collect_manifest_test_results(tests_dir: Path, manifest_path: Path) -> List[TestResult]:
    """Collect only the test_N directories listed in a latest-run manifest."""
    results = []
    seen = set()
    repo_root = tests_dir.parent

    try:
        manifest_lines = manifest_path.read_text().splitlines()
    except OSError:
        return results

    for line in manifest_lines:
        raw_path = line.strip()
        if not raw_path:
            continue

        try:
            tests_dir_resolved = tests_dir.resolve()
        except OSError:
            continue

        for candidate in manifest_test_dir_candidates(raw_path, tests_dir_resolved, repo_root):
            try:
                test_dir = candidate.resolve()
            except OSError:
                continue

            if test_dir in seen or not test_dir.is_dir():
                continue

            result = make_result_from_test_dir(tests_dir_resolved, test_dir)
            if result is None:
                continue

            seen.add(test_dir)
            results.append(result)
            break

    return results


def aggregate_results(results: List[TestResult]) -> Dict[Tuple, Dict]:
    """Group identical runs and calculate averages."""
    # The aggregation boundary is TestResult.key(). That means two rows with
    # different measured delta_t values will not be combined, even if their
    # directory parameters are otherwise identical.
    grouped = defaultdict(list)

    for result in results:
        grouped[result.key()].append(result)

    aggregated = {}
    for key, group in grouped.items():
        # Count how many expected repetitions produced usable logs. The script
        # reports missing data separately from measured values.
        with_log = sum(1 for r in group if r.has_log)
        total = len(group)
        missing_pct = 100 * (total - with_log) / total if total > 0 else 0

        # Average statistics only across successful parses. Missing logs should
        # affect %err, not pull numerical averages toward zero.
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
    # Formatting is intentionally kept separate from collection and aggregation.
    # The rest of the code returns structured data; only this function decides
    # how wide columns are and how floats should be displayed.
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

    # Fixed column widths make repeated summaries easy to compare in a terminal
    # or saved text file.
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

    # Sort by the full configuration key so repeated executions produce stable
    # output even if the filesystem returns directories in a different order.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matrix-free and matrix-based test results."
    )
    parser.add_argument(
        "tests_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Path to the tests directory.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Summarize only test_N directories listed in the latest-run manifest.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every test_N directory under tests_dir.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=f"Path to the latest-run manifest. Defaults to tests_dir/{LATEST_RUN_MANIFEST}.",
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
    return parser.parse_args()


def main():
    # The optional positional argument lets the same script summarize another
    # test tree. Without it, the repository's tests/ directory is used.
    args = parse_args()
    try:
        tests_dir = resolve_tests_dir(args)
    except ValueError as exception:
        print(f"Error: {exception}")
        sys.exit(2)

    if not tests_dir.exists():
        print(f"Error: tests directory not found at {tests_dir}")
        sys.exit(1)

    manifest_path = args.manifest or (tests_dir / LATEST_RUN_MANIFEST)
    use_latest = args.latest or (not args.all and manifest_path.exists())

    print("Collecting test results...")
    if use_latest:
        results = collect_manifest_test_results(tests_dir, manifest_path)
        print(f"Using latest-run manifest: {manifest_path}")
    else:
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
