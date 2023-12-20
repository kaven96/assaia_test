#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <video_path> <polygon_path> <output_path> <gt_path>"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before proceeding."
    exit 1
fi

# Check if deps_installed file exists
deps_file="deps_installed"

if [ ! -f "$deps_file" ]; then
    echo "Dependencies not installed. Creating virtual environment and installing required packages..."

    # Create virtual environment
    python3 -m venv venv || { echo "Failed to create a virtual environment."; exit 1; }

    # Activate virtual environment
    source venv/bin/activate || { echo "Failed to activate the virtual environment."; exit 1; }

    # Install required Python libraries
    pip install -r requirements.txt || { echo "Failed to install Python libraries."; exit 1; }

    # Mark dependencies as installed
    touch "$deps_file"

    # Deactivate virtual environment
    deactivate
fi

# Assign input arguments to variables
video_path=$1
polygon_path=$2
output_path=$3
gt_intervals=$4

# Get into virtual environment
source venv/bin/activate

echo "Processing videos..."
python3 run.py "$video_path" "$polygon_path"

echo "Evaluating results"
python3 evaluate.py "$gt_intervals" final_output_result.json

# move results file to the output_path
mv final_output_result.json "$output_path"

# print resulting metrics
echo "$(cat metrics_all.txt)"

echo "Detection and evaluation completed. Results saved to: $output_path"
