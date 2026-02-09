#!/bin/bash
# Development install script for MuJoCo Python bindings.
# Builds the C++ library and installs the Python package in editable mode.
#
# Usage:
#   ./dev_install.sh              # Full build: C++ library + editable Python install
#   ./dev_install.sh -p           # Python-only: skip C++ rebuild
#   ./dev_install.sh -c           # C++-only: rebuild C++ library only
#   ./dev_install.sh --clean      # Remove all build artifacts and start fresh
#
# Environment variables:
#   MUJOCO_DEV_PREFIX   C++ install prefix (default: $HOME/.mujoco_dev)
#   BUILD_TYPE          CMake build type (default: Release)
#   PARALLEL_JOBS       Build parallelism (default: auto-detected)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────────

MUJOCO_DEV_PREFIX="${MUJOCO_DEV_PREFIX:-$HOME/.mujoco_dev}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

BUILD_DIR="$REPO_ROOT/build"

# ── Parse flags ────────────────────────────────────────────────────────────────

DO_CPP=true
DO_PYTHON=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-only|-p)
      DO_CPP=false
      DO_PYTHON=true
      shift
      ;;
    --cpp-only|-c)
      DO_CPP=true
      DO_PYTHON=false
      shift
      ;;
    --clean)
      echo "==> Cleaning build artifacts..."
      rm -rf "$BUILD_DIR"
      rm -rf "$MUJOCO_DEV_PREFIX"
      rm -rf "$REPO_ROOT"/python/mujoco/include
      rm -rf "$REPO_ROOT"/python/mujoco/plugin
      rm -rf "$REPO_ROOT"/python/mujoco/MuJoCo_*
      rm -f  "$REPO_ROOT"/python/mujoco/cmake
      rm -f  "$REPO_ROOT"/python/mujoco/simulate
      rm -f  "$REPO_ROOT"/python/mujoco/enum_traits.h
      rm -f  "$REPO_ROOT"/python/mujoco/function_traits.h
      rm -f  "$REPO_ROOT"/python/mujoco/specs.cc.inc
      rm -rf "$REPO_ROOT"/python/*.egg-info
      echo "   Done."
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      echo "Usage: $0 [--python-only|-p] [--cpp-only|-c] [--clean]" >&2
      exit 1
      ;;
  esac
done

# ── Step 1: Build & install C++ library ────────────────────────────────────────

if $DO_CPP; then
  echo "==> Building MuJoCo C++ library (${BUILD_TYPE})..."
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  cmake --build "$BUILD_DIR" -j"$PARALLEL_JOBS"

  echo "==> Installing to $MUJOCO_DEV_PREFIX..."
  cmake --install "$BUILD_DIR" --prefix "$MUJOCO_DEV_PREFIX"
fi

# ── Step 2: Editable Python install ───────────────────────────────────────────

if $DO_PYTHON; then
  # Create symlinks that setup.py expects (normally created by make_sdist.sh)
  echo "==> Creating symlinks for source-tree build..."
  ln -sfn ../../cmake     "$REPO_ROOT/python/mujoco/cmake"
  ln -sfn ../../simulate  "$REPO_ROOT/python/mujoco/simulate"

  # Clean directories that setup.py creates with os.makedirs() (no exist_ok)
  echo "==> Cleaning setup.py artifact directories..."
  rm -rf "$REPO_ROOT/python/mujoco/plugin/"
  rm -rf "$REPO_ROOT/python/mujoco/include/"
  rm -rf "$REPO_ROOT/python/mujoco/MuJoCo_"*

  echo "==> Ensuring build dependencies..."
  uv pip install setuptools numpy absl-py

  # Pre-generate codegen files so CMake finds them in the source tree.
  # (CMake's custom_command fallback only works at build time, but the
  # generate step needs the files to exist.)
  CODEGEN_DIR="$REPO_ROOT/python/mujoco/codegen"
  CODEGEN_OUT="$REPO_ROOT/python/mujoco"
  echo "==> Generating codegen files..."
  PYTHONPATH="$REPO_ROOT/python/mujoco" \
    python "$CODEGEN_DIR/generate_enum_traits.py" > "$CODEGEN_OUT/enum_traits.h"
  PYTHONPATH="$REPO_ROOT/python/mujoco" \
    python "$CODEGEN_DIR/generate_function_traits.py" > "$CODEGEN_OUT/function_traits.h"
  PYTHONPATH="$REPO_ROOT/python/mujoco" \
    python "$CODEGEN_DIR/generate_spec_bindings.py" > "$CODEGEN_OUT/specs.cc.inc"

  echo "==> Installing Python package (editable)..."
  MUJOCO_PATH="$MUJOCO_DEV_PREFIX" \
  MUJOCO_PLUGIN_PATH="$REPO_ROOT/build/lib" \
    uv pip install -e "$REPO_ROOT/python/[sysid]" --no-build-isolation

  echo "==> Verifying installation..."
  python -c "import mujoco; print(f'mujoco {mujoco.__version__} OK')"
fi

echo "==> All done."
