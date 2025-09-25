#!/bin/bash

# Unified pipeline runner
# Usage: ./pipeline.sh <image_path> [-t "optional, tags"]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Error: No image path provided"
    echo "Usage: ./pipeline.sh <image_path> [-t \"optional, tags\"]"
    exit 1
fi

IMAGE_PATH="$1"
shift || true

OPTIONAL_TAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--optional-tags)
            shift
            OPTIONAL_TAGS="${1:-}"
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./pipeline.sh <image_path> [-t \"optional, tags\"]"
            exit 1
            ;;
    esac
    shift || true
done

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: File '$IMAGE_PATH' does not exist"
    exit 1
fi

# Prefer local venv if present
PYTHON_BIN="python3"
if [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif [ -x "${SCRIPT_DIR}/../.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/../.venv/bin/python"
fi

cd "$SCRIPT_DIR" || exit 1

CMD=("$PYTHON_BIN" "pipeline.py" "$IMAGE_PATH")
if [ -n "$OPTIONAL_TAGS" ]; then
    CMD+=("-t" "$OPTIONAL_TAGS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


