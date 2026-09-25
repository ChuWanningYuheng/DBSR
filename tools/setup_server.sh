#!/usr/bin/env bash
# Install pydbsr on a Linux server into a directory on a large disk.
#
#   bash setup_server.sh /path/on/big/disk/pydbsr_calc
#
# Everything (Miniforge, the conda environment, caches, temporary files,
# calculations) goes into that directory, so the (small) home directory is not
# filled.  Afterwards:   source /path/on/big/disk/pydbsr_calc/env.sh
set -euo pipefail

ROOT="${1:?usage: bash setup_server.sh /big/disk/pydbsr_calc}"
REPO="${PYDBSR_REPO:-https://github.com/ChuWanningYuheng/DBSR.git}"
mkdir -p "$ROOT"/{runs,tmp,cache/pip,cache/conda}
ROOT="$(cd "$ROOT" && pwd)"
echo "== pydbsr will be installed in $ROOT"
df -h "$ROOT" | tail -1

export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" CONDA_PKGS_DIRS="$ROOT/cache/conda"

# ---------------------------------------------------------------- Miniforge (conda)
if [ ! -x "$ROOT/miniforge3/bin/conda" ]; then
  arch="$(uname -m)"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  echo "== downloading $url"
  curl -fsSL -o "$ROOT/tmp/miniforge.sh" "$url" || wget -q -O "$ROOT/tmp/miniforge.sh" "$url"
  bash "$ROOT/tmp/miniforge.sh" -b -p "$ROOT/miniforge3"
  rm -f "$ROOT/tmp/miniforge.sh"
fi
# shellcheck disable=SC1091
source "$ROOT/miniforge3/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- environment
if [ ! -d "$ROOT/env" ]; then
  conda create -y -p "$ROOT/env" -c conda-forge python=3.11 \
      fortran-compiler c-compiler cmake ninja git "libblas=*=*openblas" liblapack openblas \
      numpy scipy mpmath matplotlib openpyxl pytest ipykernel jupyterlab scikit-build-core
fi
conda activate "$ROOT/env"
python -c "import jupyterlab, matplotlib, openpyxl" 2>/dev/null \
  || conda install -y -c conda-forge jupyterlab matplotlib openpyxl       # older environments
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# ---------------------------------------------------------------- pydbsr (re-running this script updates it)
CMAKE_BUILD_PARALLEL_LEVEL=16 pip install --no-build-isolation --force-reinstall --no-deps -v "git+$REPO" \
    > "$ROOT/install.log" 2>&1 \
  || { echo "!! build failed, see $ROOT/install.log (last lines below)"; tail -40 "$ROOT/install.log"; exit 1; }
pydbsr info

# sources: examples, notebook, stress test
if [ -d "$ROOT/DBSR_src/.git" ]; then git -C "$ROOT/DBSR_src" pull -q; else git clone -q "$REPO" "$ROOT/DBSR_src"; fi
cp -n "$ROOT/DBSR_src/examples/xe_plus_wang2019.ipynb" "$ROOT/runs/" 2>/dev/null || true

# Jupyter kernel (so that the existing notebook can use this environment)
python -m ipykernel install --user --name pydbsr --display-name "Python (pydbsr)" || true

# ---------------------------------------------------------------- env.sh
cat > "$ROOT/env.sh" <<EOF
# source $ROOT/env.sh
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/env"
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" CONDA_PKGS_DIRS="$ROOT/cache/conda"
export PYDBSR_CACHE="$ROOT/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # pydbsr sets threads per job
cd "$ROOT/runs"
EOF
echo
echo "== done.  Use:   source $ROOT/env.sh"
echo "   runs go to    $ROOT/runs"
echo "   notebook      $ROOT/runs/xe_plus_wang2019.ipynb  (kernel 'Python (pydbsr)')"
