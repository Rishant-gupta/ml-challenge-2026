#!/usr/bin/env bash
# ML Challenge 2026: Convenient JupyterLab Launcher

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.jupyter"
export JUPYTER_DATA_DIR="$PROJECT_DIR/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.jupyter/runtime"
export IPYTHONDIR="$PROJECT_DIR/.ipython"

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"

echo "=========================================================="
echo "🚀 Starting JupyterLab for ML Challenge 2026..."
echo "📁 Notebooks available in: notebooks/"
echo "=========================================================="

"$PROJECT_DIR/venv/bin/jupyter" lab --no-browser --ip=0.0.0.0 --port=8888 "$@"
