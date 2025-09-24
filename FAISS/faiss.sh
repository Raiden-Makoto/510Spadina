#!/bin/bash

# Script to run queensquay.py with proper virtual environment activation
# Usage: ./run_queensquay.sh "your query here"
# Example: ./run_queensquay.sh "cat ears, green eyes"

# Check if query argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"your query here\""
    echo "Example: $0 \"cat ears, green eyes\""
    exit 1
fi

QUERY="$1"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
else
    # Activate virtual environment
    source .venv/bin/activate
fi

# Run the queensquay.py script with the query
echo "Starting Genshin Character Matcher..."
echo "Query: $QUERY"
echo "Note: You'll need to select 1-3 when the images appear."
python faiss.py "$QUERY"

# Deactivate virtual environment when done
deactivate
