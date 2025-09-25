#!/bin/bash

# Style Transfer Runner Script
# Usage: ./styletransfer.sh <image_path> [-t "optional, tags"]
# Example: ./styletransfer.sh ../GenshinCharacters/Fischl.png -t "blonde, green eyes"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Error: No image path provided"
    echo "Usage: ./styletransfer.sh <image_path> [-t \"optional, tags\"]"
    echo "Example: ./styletransfer.sh ../GenshinCharacters/Fischl.png -t \"blonde, green eyes\""
    exit 1
fi

IMAGE_PATH="${1}"
shift

OPTIONAL_TAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--optional-tags)
            shift
            OPTIONAL_TAGS="$1"
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./styletransfer.sh <image_path> [-t \"optional, tags\"]"
            exit 1
            ;;
    esac
    shift
done

# Check if the file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: File '$IMAGE_PATH' does not exist"
    exit 1
fi

# Check if the file is an image
if ! file "$IMAGE_PATH" | grep -q "image"; then
    echo "Error: '$IMAGE_PATH' is not a valid image file"
    exit 1
fi

echo "Ensure NVIDIA (CUDA) or Apple Silicon (MPS) GPU is available."
echo "It is not recommended to run this script on only CPU."
echo "If using a venv, run 'source .venv/bin/activate' first or let this script auto-detect it."
echo "Install dependencies with 'pip install -r requirements.txt' if not already installed."
echo "Ensure at least 8GB of VRAM is available."
echo "----------------------------------------------------------"
echo "Starting style transfer for: $IMAGE_PATH"
echo "This may take a few minutes..."
echo ""

# Choose Python (prefer local venv if present)
PYTHON_BIN="python3"
if [ -x "${SCRIPT_DIR}/../.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/../.venv/bin/python"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
fi

# Run the Python script from its directory so relative assets resolve
cd "${SCRIPT_DIR}" || exit 1

if [ -n "$OPTIONAL_TAGS" ]; then
    "$PYTHON_BIN" styletransfer.py "$IMAGE_PATH" -t "$OPTIONAL_TAGS"
else
    "$PYTHON_BIN" styletransfer.py "$IMAGE_PATH"
fi

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Style transfer completed successfully!"
else
    echo ""
    echo "❌ Style transfer failed!"
    exit 1
fi
