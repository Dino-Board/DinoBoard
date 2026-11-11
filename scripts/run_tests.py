#!/usr/bin/env python3
"""Run DinoBoard pytest files in parallel without requiring pytest-xdist."""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TestResult:
    index: int
    path: Path
    returncode: int
    duration: float
    output: str


def _discover(paths: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"test path does not exist: {raw}")
        if path.is_file():
            if path.name.startswith("test_") and path.suffix == ".py":
                discovered.append(path)
            continue
        for child in sorted(path.rglob("test_*.py")):
            if child.is_file():
                discovered.append(child)
    unique = sorted({p.resolve() for p in discovered})
    if not unique:
        raise RuntimeError("no pytest files found")
    return unique


def _worker(
    worker_id: int,
    work_q: queue.Queue[tuple[int, Path]],
    results: list[TestResult],
    results_lock: threading.Lock,
    args: argparse.Namespace,
    model_root: Path,
) -> None:
    model_dir = model_root / f"worker_{worker_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DINOBOARD_TEST_MODEL_DIR"] = str(model_dir)
    if args.force_cpu_threads:
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

    while True:
        try:
            index, path = work_q.get_nowait()
        except queue.Empty:
            return
        rel = path.relative_to(ROOT)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(rel),
            *args.pytest_args,
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        duration = time.perf_counter() - start
        with results_lock:
            results.append(TestResult(
                index=index,
                path=rel,
                returncode=proc.returncode,
                duration=duration,
                output=proc.stdout,
            ))
        work_q.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest files in parallel with per-worker model caches.")
    parser.add_argument(
        "paths", nargs="*", default=["tests"],
        help="Test files or directories (default: tests)")
    parser.add_argument(
        "-j", "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel pytest subprocesses (default: half of CPU count)")
    parser.add_argument(
        "--serial", action="store_true",
        help="Run one pytest process over all paths, matching legacy behavior.")
    parser.add_argument(
        "--keep-model-cache", action="store_true",
        help="Keep the temporary per-worker ONNX model cache for inspection.")
    parser.add_argument(
        "--force-cpu-threads", action="store_true", default=True,
        help="Set common BLAS/OpenMP thread env vars to 1 in worker processes.")
    parser.add_argument(
        "--no-force-cpu-threads", action="store_false",
        dest="force_cpu_threads",
        help="Do not modify BLAS/OpenMP thread env vars.")
    parser.add_argument(
        "--pytest-args", nargs=argparse.REMAINDER, default=["-q"],
        help="Arguments passed after each pytest file. Default: -q. "
             "Use --pytest-args -q -x to stop a worker on first failure.")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    tests = _discover(args.paths)
    if args.serial or args.workers == 1:
        env = os.environ.copy()
        if "DINOBOARD_TEST_MODEL_DIR" not in env:
            env["DINOBOARD_TEST_MODEL_DIR"] = "/tmp/dinoboard_test_models"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *[str(p.relative_to(ROOT)) for p in tests],
            *args.pytest_args,
        ]
        return subprocess.call(cmd, cwd=ROOT, env=env)

    workers = min(args.workers, len(tests))
    model_root = Path(tempfile.mkdtemp(prefix="dinoboard_test_models_"))
    work_q: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in enumerate(tests):
        work_q.put(item)

    results: list[TestResult] = []
    results_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(i, work_q, results, results_lock, args, model_root),
            daemon=False,
        )
        for i in range(workers)
    ]

    print(
        f"Running {len(tests)} pytest files with {workers} workers "
        f"(model cache: {model_root})",
        flush=True,
    )
    start_all = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    total_duration = time.perf_counter() - start_all

    results.sort(key=lambda r: r.index)
    failed = [r for r in results if r.returncode != 0]
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"[{status}] {result.path} ({result.duration:.1f}s)", flush=True)
        if result.returncode != 0:
            print(result.output.rstrip(), flush=True)

    print(
        f"\nFinished {len(results)} files in {total_duration:.1f}s: "
        f"{len(results) - len(failed)} passed, {len(failed)} failed",
        flush=True,
    )
    if not args.keep_model_cache:
        shutil.rmtree(model_root, ignore_errors=True)
    else:
        print(f"Kept model cache: {model_root}", flush=True)
    if failed:
        print("Failed files:", flush=True)
        for result in failed:
            print(f"  {result.path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
