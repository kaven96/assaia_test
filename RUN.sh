#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <video_path> <polygon_path> <output_path>"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before proceeding."
    exit 1
fi

# Read required packages from requirements.txt
required_packages=($(cat requirements.txt))

# Check if required Python packages are installed
missing_packages=()
for package in "${required_packages[@]}"; do
    if ! pip list | grep -q "$package"; then
        missing_packages+=("$package")
    fi
done

# If at least one package is missing, create a virtual environment and install required packages
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "At least one required Python package is missing. Creating virtual environment and installing required packages..."

    # Create virtual environment
    python3 -m venv venv || { echo "Failed to create virtual environment."; exit 1; }

    # Activate virtual environment
    source venv/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }

    # Install required Python libraries
    pip install -r requirements.txt || { echo "Failed to install Python libraries."; exit 1; }

    # Deactivate virtual environment
    deactivate
fi

# Assign input arguments to variables
video_path=$1
polygon_path=$2
output_path=$3

# Run the detection script
python3 detect5.py "$video_path" "$polygon_path"

# Run the evaluation script
python3 evaluate.py 'time_intervals.json' 'output_result.json' "$video_path"  

# Move the final output to the specified output path
mv 'output_result.json' "$output_path"

echo "Detection and evaluation completed. Results saved to: $output_path"
