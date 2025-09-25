#!/bin/bash

# Script to run dino.py for Human to Anime Feature Matching

# Check if image path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_test_image>"
    echo "Example: $0 ../GenshinCharacters/Hutao.png"
    exit 1
fi

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "../.venv" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the Python script with the provided image path
echo "Running Human to Anime Feature Matcher..."
echo "Test image: $1"
python dino.py "$1"

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
