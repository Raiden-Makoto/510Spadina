#!/bin/bash

# Style Transfer Runner Script
# Usage: ./run_style_transfer.sh <image_path>
# Example: ./run_style_transfer.sh ../GenshinCharacters/Fischl.png

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No image path provided"
    echo "Usage: ./run_style_transfer.sh <image_path>"
    echo "Example: ./run_style_transfer.sh ../GenshinCharacters/Fischl.png"
    exit 1
fi

# Get the image path from argument
IMAGE_PATH="$1"

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

echo "Please ensure you are running on a machine with an NVIDIA GPU for CUDA support or Apple Silicon GPU for MPS support"
echo "It is not recommended to run this script on only CPU"
echo "Please ensure you have activated the virtual environment"
echo "Run 'source .venv/bin/activate' to activate the virtual environment"
echo "Please ensure you have installed the required dependencies"
echo "Run 'pip install -r requirements.txt' to install the dependencies"
echo "Please ensure that you have at least 8GB of VRAM available"
echo "----------------------------------------------------------"
echo "Starting style transfer for: $IMAGE_PATH"
echo "This may take a few minutes..."
echo ""

# Run the Python script
python3 styletransfer.py "$IMAGE_PATH"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Style transfer completed successfully!"
else
    echo ""
    echo "❌ Style transfer failed!"
    exit 1
fi
