#!/usr/bin/env python3
"""@file summarize_tests.py
@brief Summarize extensive test results into a formatted table.

@details
The script treats the on-disk test tree as the source of the experiment
configuration, and the log.txt files as the source of the measured quantities.
Each test_N directory becomes one TestResult. Results with the same
configuration key are then averaged into one table row.

Table columns:
  solver problem fe_deg n_ranks n_threads simd n_add_ref delta_t
  adj_tol err it_n total_t %conv %overtime %fail

Identical runs are aggregated with averages and percentages.
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
    "adj_tol",
    "rel_t_step",
    "it_n",
    "err",
    "total_t",
    "l2_error",
    "h1_error",
    "linfty_error",
    "converged",
]

LEGACY_FILE_COLUMNS = [
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
ATTEMPT_MANIFEST_PREFIX = "ATTEMPT"


def legacy_fe_degree_for_solver(solver: str) -> int:
    """Return the old hard-coded degree for pre-fe_deg result trees."""
    # New results store fe_deg in the directory path. This fallback keeps old
    # tests/{solver}/{problem}/{n_add_ref}/... folders readable without mixing
    # matrix-based degree-2 runs and matrix-free degree-4 runs under one label.
    return 4 if solver == "matrix_free" else 2


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
    fe_deg: int
    n_ranks: int
    n_threads: int
    simd: int
    n_additional_refinements: int
    delta_t: float

    max_iter: Optional[int] = None
    requested_tol: Optional[float] = None
    adj_tol: Optional[float] = None
    it_n: Optional[float] = None
    err: Optional[float] = None
    converged: Optional[float] = None
    total_t: Optional[float] = None
    has_log: bool = False
    status: Optional[str] = None
    output_dirs: Tuple[Path, ...] = ()
    log_parse_error: Optional[str] = None

    def key(self) -> Tuple:
        """Return key for grouping repeated runs of the same configuration."""
        # test_N is deliberately excluded. Repeated test_N directories under
        # the same configuration are samples of the same experiment.
        if self.requested_tol is not None:
            tolerance_key = ("requested_tol", self.requested_tol)
        elif self.adj_tol is not None:
            tolerance_key = ("adj_tol", self.adj_tol)
        else:
            tolerance_key = ("unknown_tol", 0.0)

        max_iter_key = self.max_iter if self.max_iter is not None else -1

        return (self.solver, self.problem, self.fe_deg, self.n_ranks,
                self.n_threads, self.simd, self.n_additional_refinements,
                self.delta_t, max_iter_key, tolerance_key)


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
    if header == FILE_COLUMNS:
        return header

    # Older logs wrote the solver tolerance factor as "tol". That is not the
    # adjusted absolute tolerance requested by the current summary, so the
    # parser keeps it under a different semantic name instead of pretending it
    # is adj_tol.
    if header == LEGACY_FILE_COLUMNS:
        return [
            "delta_t",
            "max_iter",
            "tol_factor",
            "rel_t_step",
            "it_n",
            "err",
            "total_t",
            "l2_error",
            "h1_error",
            "linfty_error",
            "converged",
        ]

    else:
        raise ValueError(
            "Unexpected log header. Expected "
            + " ".join(FILE_COLUMNS)
            + " or "
            + " ".join(LEGACY_FILE_COLUMNS)
            + ", got "
            + " ".join(header)
        )


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
    current logs store the adjusted absolute tolerance, so convergence is
    evaluated as final residual < adjusted tolerance. Legacy logs fall back to
    the stored SolverControl convergence flag because they did not record the
    RHS norm needed to reconstruct the adjusted tolerance.
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
    time_steps = {}  # rel_t_step -> list of (it_n, err, adj_tol)
    delta_t = None
    max_iter = None
    requested_tol = None
    total_t = None
    logged_converged = None

    for line in data_lines:
        # A malformed row should stop the summary instead of being interpreted
        # as "this test has no log". The current C++ logger writes a fixed set
        # of whitespace-separated columns, so anything else is a real mismatch.
        parts = parse_data_line(line, len(header))
        row = dict(zip(header, parts))

        # The log writer already chose scientific or fixed formatting. Once the
        # columns are known, the summarizer stores only typed values.
        delta_t = float(row['delta_t'])
        max_iter = int(row['max_iter'])
        rel_t_step = float(row['rel_t_step'])
        it_n = int(row['it_n'])
        err = float(row['err'])
        total_t = float(row['total_t'])
        adj_tol = float(row['adj_tol']) if 'adj_tol' in row else None
        requested_tol = float(row['tol_factor']) if 'tol_factor' in row else None

        # This flag is written by SolverControl and answers the run status
        # directly for legacy logs. Current logs also carry adj_tol, allowing
        # the summary to use the explicit user-requested residual test.
        logged_converged = int(float(row['converged']))

        if rel_t_step not in time_steps:
            time_steps[rel_t_step] = []
        time_steps[rel_t_step].append((it_n, err, adj_tol))

    if not time_steps:
        return None

    # For each representative time position, the last row is the final residual
    # checkpoint for that slice. Averaging those endpoints gives one compact
    # result per test directory.
    time_step_it_ns = []
    time_step_errs = []
    time_step_adj_tols = []

    for ts_data in time_steps.values():
        final_it_n, final_err, final_adj_tol = ts_data[-1]
        time_step_it_ns.append(final_it_n)
        time_step_errs.append(final_err)
        if final_adj_tol is not None:
            time_step_adj_tols.append(final_adj_tol)

    if len(time_step_adj_tols) == len(time_steps):
        overall_converged = int(
            all(final_err < final_adj_tol
                for _, final_err, final_adj_tol in
                (ts_data[-1] for ts_data in time_steps.values()))
        )
        avg_adj_tol = statistics.mean(time_step_adj_tols)
    else:
        overall_converged = logged_converged
        avg_adj_tol = None

    return {
        'max_iter': max_iter,
        'requested_tol': requested_tol,
        'adj_tol': avg_adj_tol,
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

    if len(relative_parts) == 8:
        solver, problem, fe_deg, n_add_ref, n_ranks, n_threads, simd, test_name = relative_parts
    elif len(relative_parts) == 7:
        # Legacy layout, before fe_deg became an explicit sweep parameter:
        # tests/{solver}/{problem}/{n_add_ref}/{n_ranks}/{n_threads}/{simd}/test_N
        solver, problem, n_add_ref, n_ranks, n_threads, simd, test_name = relative_parts
        fe_deg = str(legacy_fe_degree_for_solver(solver))
    else:
        return None

    if not test_name.startswith('test_'):
        return None

    try:
        fe_degree = int(fe_deg)
        n_additional_refinements = int(n_add_ref)
        n_ranks_value = int(n_ranks)
        n_threads_value = int(n_threads)
        simd_value = int(simd)
    except ValueError:
        return None

    result = TestResult(
        solver=solver,
        problem=problem,
        fe_deg=fe_degree,
        n_ranks=n_ranks_value,
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
            result.max_iter = parsed['max_iter']
            result.requested_tol = parsed['requested_tol']
            result.adj_tol = parsed['adj_tol']
            result.it_n = parsed['it_n']
            result.err = parsed['err']
            result.converged = parsed['converged']
            result.total_t = parsed['total_t']
            result.has_log = True
            result.status = "ok"

    return result


def parse_attempt_manifest_line(line: str) -> Optional[Dict[str, str]]:
    """Parse one structured attempt line from run_extensive_tests.sh."""
    parts = line.strip().split()
    if not parts or parts[0] != ATTEMPT_MANIFEST_PREFIX:
        return None

    record = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        record[key] = value

    return record


def output_dir_candidates_from_attempt(record: Dict[str, str],
                                       tests_dir: Path,
                                       repo_root: Path) -> List[Path]:
    """Return all output directories named by a structured manifest record."""
    raw_output_dirs = record.get("output_dirs", "")
    candidates = []

    for raw_path in raw_output_dirs.split(","):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        candidates.extend(manifest_test_dir_candidates(raw_path, tests_dir, repo_root))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


def make_result_from_attempt_record(tests_dir: Path,
                                    record: Dict[str, str]) -> Optional[TestResult]:
    """Build one TestResult from an ATTEMPT manifest record."""
    try:
        result = TestResult(
            solver=record["solver"],
            problem=record["problem"],
            fe_deg=int(record["fe_deg"]),
            n_ranks=int(record["n_ranks"]),
            n_threads=int(record["n_threads"]),
            simd=int(record["simd"]),
            n_additional_refinements=int(record["n_add_ref"]),
            delta_t=float(record["delta_t"]),
            max_iter=int(record["max_iters"]),
            requested_tol=float(record["tol"]),
            status=record["status"],
        )
    except (KeyError, ValueError):
        return None

    try:
        tests_dir_resolved = tests_dir.resolve()
    except OSError:
        tests_dir_resolved = tests_dir

    repo_root = tests_dir_resolved.parent
    output_dirs = []

    for candidate in output_dir_candidates_from_attempt(record, tests_dir_resolved, repo_root):
        try:
            test_dir = candidate.resolve()
        except OSError:
            continue

        if not test_dir.is_dir():
            continue

        output_dirs.append(test_dir)
        log_file = test_dir / "log.txt"
        if not log_file.exists():
            continue

        try:
            parsed = parse_log_file(log_file)
        except ValueError as exception:
            # A timeout or failing solver may leave a partial log. Preserve the
            # attempt and let its runner status drive classification instead of
            # losing the denominator for the percentage columns.
            if result.status in {"overtime", "fail"}:
                result.log_parse_error = str(exception)
                continue
            raise

        if parsed:
            # For structured latest-run manifests the command-line delta_t is
            # kept as the grouping key, because failed attempts may never reach
            # the point where the solver can report a problem-default delta_t.
            result.adj_tol = parsed['adj_tol']
            result.it_n = parsed['it_n']
            result.err = parsed['err']
            result.converged = parsed['converged']
            result.total_t = parsed['total_t']
            result.has_log = True
            break

    result.output_dirs = tuple(output_dirs)
    return result


def collect_test_results(tests_dir: Path) -> List[TestResult]:
    """Collect all test results from directory structure."""
    # The directory path encodes the parameters that were used to launch the
    # run. Recursive traversal supports both the new fe_deg-aware layout and
    # the legacy layout parsed by make_result_from_test_dir().
    results = []

    # New directory structure:
    # tests/{solver}/{problem}/{fe_deg}/{n_add_ref}/{n_ranks}/{n_threads}/{simd}/test_N/log.txt
    for test_dir in tests_dir.rglob("test_*"):
        if not test_dir.is_dir() or not test_dir.name.startswith('test_'):
            continue

        result = make_result_from_test_dir(tests_dir, test_dir)
        if result is not None:
            results.append(result)

    return results


def collect_manifest_test_results(tests_dir: Path, manifest_path: Path) -> List[TestResult]:
    """Collect only the test_N directories listed in a latest-run manifest."""
    structured_results = []
    legacy_results = []
    seen = set()
    repo_root = tests_dir.parent

    try:
        manifest_lines = manifest_path.read_text().splitlines()
    except OSError:
        return structured_results

    for line in manifest_lines:
        raw_path = line.strip()
        if not raw_path:
            continue

        attempt_record = parse_attempt_manifest_line(raw_path)
        if attempt_record is not None:
            result = make_result_from_attempt_record(tests_dir, attempt_record)
            if result is not None:
                structured_results.append(result)
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
            legacy_results.append(result)
            break

    # New manifests contain both shell-written ATTEMPT records and legacy
    # solver-written bare paths. The ATTEMPT records are authoritative because
    # they include failures and timeouts that may not have a completed log.
    if structured_results:
        return structured_results

    return legacy_results


def classify_result(result: TestResult) -> str:
    """Classify one attempt into exactly one summary bucket."""
    if result.status == "fail":
        return "fail"

    if result.status == "overtime":
        return "overtime"

    # A nominally successful command without a final parseable log violates the
    # producer contract. Count it as a failure so percentages still sum to 100%
    # and the missing evidence is visible.
    if not result.has_log:
        return "fail"

    return "conv" if bool(result.converged) else "overtime"


def mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def aggregate_results(results: List[TestResult]) -> Dict[Tuple, Dict]:
    """Group identical runs and calculate averages and status percentages."""
    # The aggregation boundary is TestResult.key(). That means two rows with
    # different requested tolerances or max-iteration limits will not be
    # combined, even though max_iter is not displayed in the compact table.
    grouped = defaultdict(list)

    for result in results:
        grouped[result.key()].append(result)

    aggregated = {}
    for key, group in grouped.items():
        total = len(group)
        classifications = [classify_result(r) for r in group]
        conv_count = classifications.count("conv")
        overtime_count = classifications.count("overtime")
        fail_count = classifications.count("fail")

        # Numerical averages are taken from non-failing attempts with usable
        # logs. A failed run may have partial output, but mixing it into timing
        # or residual averages would make the table look more precise than the
        # failed execution allows.
        measured_runs = [
            r for r, classification in zip(group, classifications)
            if classification != "fail" and r.has_log
        ]

        aggregated[key] = {
            'adj_tol': mean_or_none([r.adj_tol for r in measured_runs if r.adj_tol is not None]),
            'it_n': mean_or_none([r.it_n for r in measured_runs if r.it_n is not None]),
            'err': mean_or_none([r.err for r in measured_runs if r.err is not None]),
            'total_t': mean_or_none([r.total_t for r in measured_runs if r.total_t is not None]),
            'conv_pct': 100 * conv_count / total if total > 0 else 0,
            'overtime_pct': 100 * overtime_count / total if total > 0 else 0,
            'fail_pct': 100 * fail_count / total if total > 0 else 0,
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
        'solver': 2,
        'problem': 9,
        'fe_deg': 6,
        'n_ranks': 5,
        'n_threads': 7,
        'simd': 1,
        'n_add_ref': 3,
        'delta_t': 6,
        'adj_tol': 10,
        'err': 9,
        'it_n': 5,
        'total_t': 7,
        'conv_pct': 6,
        'overtime_pct': 10,
        'fail_pct': 6,
    }

    def compact_solver_name(solver: str) -> str:
        if solver == "matrix_based":
            return "mb"
        if solver == "matrix_free":
            return "mf"
        return solver

    def table_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    def format_optional(value: Optional[float], width: int, fmt: str) -> str:
        if value is None:
            return f"{'n/a':>{width}}"
        return f"{value:{fmt}}"

    def format_pct(value: float, width: int) -> str:
        return f"{value:>{width - 1}.1f}%"

    # Short labels keep the table below a typical 100-column terminal while
    # vertical separators keep columns readable after the widths are tightened.
    header = table_row([
        f"{'sv':<{col_widths['solver']}}",
        f"{'problem':<{col_widths['problem']}}",
        f"{'fe_deg':>{col_widths['fe_deg']}}",
        f"{'ranks':>{col_widths['n_ranks']}}",
        f"{'threads':>{col_widths['n_threads']}}",
        f"{'s':>{col_widths['simd']}}",
        f"{'ref':>{col_widths['n_add_ref']}}",
        f"{'dt':>{col_widths['delta_t']}}",
        f"{'adj_tol':>{col_widths['adj_tol']}}",
        f"{'err':>{col_widths['err']}}",
        f"{'it':>{col_widths['it_n']}}",
        f"{'time':>{col_widths['total_t']}}",
        f"{'%conv':>{col_widths['conv_pct']}}",
        f"{'%overtime':>{col_widths['overtime_pct']}}",
        f"{'%fail':>{col_widths['fail_pct']}}",
    ])
    separator = "-" * len(header)

    lines.append(header)
    lines.append(separator)

    # Sort by the full configuration key so repeated executions produce stable
    # output even if the filesystem returns directories in a different order.
    for key in sorted(aggregated.keys()):
        solver, problem, fe_deg, n_ranks, n_threads, simd, n_add_ref, delta_t, _max_iter, _tolerance_key = key
        stats = aggregated[key]

        line = table_row([
            f"{compact_solver_name(solver):<{col_widths['solver']}}",
            f"{problem:<{col_widths['problem']}}",
            f"{fe_deg:>{col_widths['fe_deg']}}",
            f"{n_ranks:>{col_widths['n_ranks']}}",
            f"{n_threads:>{col_widths['n_threads']}}",
            f"{simd:>{col_widths['simd']}}",
            f"{n_add_ref:>{col_widths['n_add_ref']}}",
            f"{delta_t:>{col_widths['delta_t']}.1g}",
            format_optional(stats['adj_tol'], col_widths['adj_tol'], f">{col_widths['adj_tol']}.2e"),
            format_optional(stats['err'], col_widths['err'], f">{col_widths['err']}.2e"),
            format_optional(stats['it_n'], col_widths['it_n'], f">{col_widths['it_n']}.1f"),
            format_optional(stats['total_t'], col_widths['total_t'], f">{col_widths['total_t']}.2f"),
            format_pct(stats['conv_pct'], col_widths['conv_pct']),
            format_pct(stats['overtime_pct'], col_widths['overtime_pct']),
            format_pct(stats['fail_pct'], col_widths['fail_pct']),
        ])
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

    table = format_table(aggregated)
    banner_width = max(len(line) for line in table.splitlines())

    print("\n" + "=" * banner_width)
    print("TEST SUMMARY")
    print("=" * banner_width)
    print(table)
    print("=" * banner_width)

    return 0


if __name__ == "__main__":
    sys.exit(main())
