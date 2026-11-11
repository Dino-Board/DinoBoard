# Source this before GPU selfplay / eval to expose the ONNX Runtime GPU
# execution provider's CUDA dependencies:
#
#     source scripts/gpu_env.sh
#
# It points LD_LIBRARY_PATH at (1) the bundled ORT GPU libs under third_party/
# and (2) the CUDA / cuDNN .so files that ship inside the torch wheel's
# nvidia-* pip packages. Run it from the repo root with your training venv
# active (the `python` on PATH must be the one that has the nvidia-* packages).
# Safe to source more than once.
_ort_lib=$(ls -d "$PWD"/third_party/onnxruntime-linux-x64-gpu-*/lib 2>/dev/null | head -1)

# Resolve a python that actually has the nvidia-* packages. Prefer an active
# venv, then an in-repo training venv, then whatever 'python'/'python3' is on
# PATH — so sourcing this works even without the venv activated.
_py=""
for _cand in "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
             "$PWD"/.venv-train-cpu/bin/python \
             "$PWD"/.venv-train/bin/python; do
  if [ -n "$_cand" ] && [ -x "$_cand" ]; then _py="$_cand"; break; fi
done
[ -z "$_py" ] && _py=$(command -v python || command -v python3)

_nv_libs=$("$_py" -c "import os,glob,nvidia;b=list(nvidia.__path__)[0];print(':'.join(sorted({os.path.dirname(p) for p in glob.glob(b+'/*/lib/*.so*')})))" 2>/dev/null)
# Optional: TensorRT EP ("tensorrt" provider) — pip install tensorrt-cu13-libs==10.*
_trt_libs=$("$_py" -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null)
[ -n "$_trt_libs" ] && _nv_libs="${_nv_libs}:${_trt_libs}"

if [ -z "$_ort_lib" ]; then
  echo "gpu_env.sh: WARNING — no third_party/onnxruntime-linux-x64-gpu-*/lib found" >&2
fi
if [ -z "$_nv_libs" ]; then
  echo "gpu_env.sh: WARNING — no nvidia-* CUDA libs found for the current 'python' (wrong venv?)" >&2
fi

export LD_LIBRARY_PATH="${_ort_lib}:${_nv_libs}:${LD_LIBRARY_PATH}"
unset _ort_lib _nv_libs _trt_libs _py _cand
