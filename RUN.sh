#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <video_path> <polygon_path> <output_path>"
    exit 1
fi

# Assign input arguments to variables
video_path=$1
polygon_path=$2
output_path=$3

# Run the detection script
python detect.py "$video_path" "$polygon_path"

# Run the evaluation script
python evaluate.py 'time_intervals.json' 'output_result.json' "$video_path"

# Move the final output to the specified output path
mv 'output_result.json' "$output_path"

echo "Detection and evaluation completed. Results saved to: $output_path"
