import json
import os
import shutil
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "unix":
            for ext in self.extensions:
                ext.extra_compile_args.append("-std=c++17")
                ext.extra_compile_args.append("-O3")
        elif self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args.append("/std:c++17")
                # MSVC 14.39 can ICE on the large template-heavy game bundle
                # when optimization or LTCG is enabled on Windows. Prefer a
                # reliable debug-style extension build over a broken build.
                ext.extra_compile_args.append("/Od")
                ext.extra_compile_args.append("/Ob0")
                ext.extra_compile_args.append("/GL-")
                ext.extra_compile_args.append("/bigobj")
                ext.extra_compile_args.append("/EHsc")
                # MSVC: need explicit UTF-8 for source files containing
                # non-ASCII string literals (e.g. Chinese game names).
                ext.extra_compile_args.append("/utf-8")
                # Avoid <Windows.h>'s `min`/`max` macros stomping std::min/max.
                ext.define_macros.append(("NOMINMAX", "1"))
                ext.define_macros.append(("WIN32_LEAN_AND_MEAN", "1"))
                ext.define_macros.append(("_CRT_SECURE_NO_WARNINGS", "1"))
        super().build_extensions()

    def copy_extensions_to_source(self):
        # Standard editable-install behavior: copy the .pyd / .so next to
        # setup.py so `import dinoboard_engine` works from the repo root.
        super().copy_extensions_to_source()
        # On Windows the loader doesn't have rpath — onnxruntime.dll must
        # sit next to the extension. Copy it from the bundled directory
        # we just linked against.
        if sys.platform == "win32" and globals().get("_BUNDLED_ORT_DLL"):
            dll_src = Path(globals()["_BUNDLED_ORT_DLL"])
            if dll_src.exists():
                target_dir = Path(self.get_finalized_command("build_ext").build_lib)
                dst = target_dir / dll_src.name
                shutil.copy2(dll_src, dst)
                # Also drop one next to setup.py for editable installs.
                shutil.copy2(dll_src, ROOT / dll_src.name)


def get_pybind_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        return ""


def _load_game_sources() -> list[str]:
    """Read games/manifest.json and return the per-game C++ source paths.

    Single source of truth for "what gets compiled into the engine"; the
    same manifest is consumed by CMakeLists.txt for the cmake build.

    Entries with `enabled: false` are skipped — set the flag to drop a
    game from the build without removing its sources from the tree. The
    optional `framework_whitelist` and `capabilities` fields are ignored
    here; they are consumed by tests/conftest.py.
    """
    manifest_path = ROOT / "games" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    out = []
    for game in manifest["games"]:
        if not game.get("enabled", True):
            continue
        gid = game["id"]
        for src in game["sources"]:
            out.append(f"games/{gid}/{src}")
    return out


sources = [
    "bindings/py_engine.cpp",
    "engine/core/action_constraint.cpp",
    "engine/core/mask_state_impl.cpp",
    "engine/core/heuristic_registry.cpp",
    "engine/search/net_mcts.cpp",
    "engine/search/tail_solver.cpp",
    "engine/infer/onnx_policy_value_evaluator.cpp",
    "engine/infer/onnx_belief_evaluator.cpp",
    "engine/infer/split_policy_value_evaluator.cpp",
    "engine/infer/batched_policy_value_evaluator.cpp",
    "engine/infer/batched_belief_evaluator.cpp",
    "engine/infer/evaluator_pool.cpp",
    "engine/runtime/selfplay_runner.cpp",
    "engine/runtime/arena_runner.cpp",
    "engine/runtime/heuristic_runner.cpp",
    "heuristic/random_heuristic.cpp",
    "heuristic/mcts_pv_heuristic.cpp",
] + _load_game_sources()

include_dirs = [
    str(ROOT),
    get_pybind_include(),
]

define_macros = []

onnx_root = os.environ.get("BOARD_AI_ONNXRUNTIME_ROOT", "")
with_onnx = os.environ.get("BOARD_AI_WITH_ONNX", "")
library_dirs = []
libraries = []

if with_onnx == "0":
    raise RuntimeError(
        "BOARD_AI_WITH_ONNX=0 is no longer supported. DinoBoard requires ONNX Runtime "
        "for selfplay/eval/web AI; there is no uniform-policy fallback. "
        "Unset BOARD_AI_WITH_ONNX and install ONNX Runtime."
    )

if with_onnx == "":
    # Auto-detect ONNX Runtime in priority order:
    #   1. Platform-matched bundled copy under
    #      `third_party/onnxruntime-<platform>-<arch>-<ver>/` — ships with
    #      the repo so a fresh `git clone` works out of the box on Linux x64
    #      and Windows x64 with no external downloads (and no cloud-firewall
    #      interference). On other platforms (Mac, Linux ARM, etc.) the
    #      bundle doesn't match and we fall through to the next candidate.
    #   2. Homebrew (Mac dev machines): /opt/homebrew or /usr/local.
    if sys.platform == "win32":
        bundle_glob = "onnxruntime-win-*"
    elif sys.platform.startswith("linux"):
        bundle_glob = "onnxruntime-linux-*"
    else:
        bundle_glob = None  # mac etc. — skip the bundle, use brew

    candidates = []
    third_party = ROOT / "third_party"
    if bundle_glob and third_party.is_dir():
        bundles = sorted(third_party.glob(bundle_glob))
        if sys.platform.startswith("linux"):
            # Prefer the GPU bundle when both CPU and GPU ORT tarballs are
            # present; CPUExecutionProvider remains available from GPU ORT,
            # while CUDAExecutionProvider is impossible from the CPU bundle.
            bundles.sort(key=lambda p: ("gpu" not in p.name, p.name))
        for p in bundles:
            if (p / "include" / "onnxruntime_c_api.h").exists():
                candidates.append(str(p))
    if sys.platform != "win32":
        candidates.extend(["/opt/homebrew", "/usr/local"])
    for candidate in candidates:
        if ((Path(candidate) / "include" / "onnxruntime_c_api.h").exists()
                or (Path(candidate) / "include" / "onnxruntime" / "onnxruntime_c_api.h").exists()):
            onnx_root = onnx_root or candidate
            with_onnx = "1"
            break
    if with_onnx != "1":
        raise RuntimeError(
            "ONNX Runtime not found. DinoBoard requires ONNX for selfplay/eval/web AI;\n"
            "there is no uniform-policy fallback — see CLAUDE.md \"No Fallbacks, No Silent Degradation\".\n"
            "Install one of:\n"
            "  - macOS:    brew install onnxruntime\n"
            "  - Linux:    drop the official tarball under third_party/onnxruntime-linux-* (auto-detected)\n"
            "  - Windows:  drop the official zip under third_party/onnxruntime-win-* (auto-detected)\n"
            "  - Custom:   export BOARD_AI_ONNXRUNTIME_ROOT=/path/to/onnxruntime"
        )

with_onnx = with_onnx == "1"

define_macros.append(("BOARD_AI_WITH_ONNX", "1"))
if not onnx_root:
    raise RuntimeError("BOARD_AI_WITH_ONNX=1 requires BOARD_AI_ONNXRUNTIME_ROOT")
include_dirs.append(str(Path(onnx_root) / "include"))
onnx_inner = Path(onnx_root) / "include" / "onnxruntime"
if onnx_inner.is_dir():
    include_dirs.append(str(onnx_inner))
library_dirs.append(str(Path(onnx_root) / "lib"))
libraries.append("onnxruntime")

# rpath so the compiled extension finds the bundled libonnxruntime.so at
# runtime without requiring LD_LIBRARY_PATH to be set by the caller.
# $ORIGIN expands to the directory of the extension itself; the .so gets
# installed to site-packages/, and the bundled ORT lives at
# <repo>/third_party/onnxruntime-*/lib relative to the build dir — but in
# editable installs the extension is dropped next to setup.py, so
# $ORIGIN/third_party/<dir>/lib is the stable relative path.
extra_link_args = []
_BUNDLED_ORT_DLL = None
if with_onnx and onnx_root:
    ort_lib_dir = Path(onnx_root) / "lib"
    if sys.platform.startswith("linux"):
        # Absolute rpath is simplest and works for editable installs. If the
        # user moves the repo after install they'll need to rebuild; that's
        # fine since pip install -e . is how everyone rebuilds anyway.
        extra_link_args.append(f"-Wl,-rpath,{ort_lib_dir}")
    elif sys.platform == "darwin":
        # macOS: rpath via clang/ld; install_name resolves at load time.
        extra_link_args.extend(["-Wl,-rpath," + str(ort_lib_dir)])
    elif sys.platform == "win32":
        # Windows has no rpath. The `.lib` is the import lib (already in
        # library_dirs above). The `.dll` must sit next to the loaded `.pyd`
        # — copied in BuildExt.copy_extensions_to_source.
        dll_path = ort_lib_dir / "onnxruntime.dll"
        if dll_path.exists():
            _BUNDLED_ORT_DLL = str(dll_path)

ext = Extension(
    name="dinoboard_engine",
    sources=[str(ROOT / s) for s in sources],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    language="c++",
    extra_compile_args=[],
    extra_link_args=extra_link_args,
)

setup(
    name="dinoboard",
    version="0.1.0",
    description="DinoBoard universal board game AI platform",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.9",
    install_requires=[
        "pybind11>=2.10",
        "fastapi>=0.100",
        "uvicorn[standard]>=0.20",
        "onnxruntime>=1.16",
        "torch>=2.0",
        "numpy>=1.24",
        "onnx>=1.14",
    ],
)
