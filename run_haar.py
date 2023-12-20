import os
import argparse
from detect import VehicleDetector
import json


def process_videos(video_list, polygons_path):
    results_dict = {}

    for video_name in video_list:
        print(f"Processing video: {video_name}")
        detector = VehicleDetector(
            video_name,
            'cars.xml',
            polygons_path
        )
        detector.load_video()
        detector.load_cascade()
        detector.load_polygons()
        detector.detect_vehicles()

        with open("output_result.json", "r") as json_file:
            current_result = json.load(json_file)

        results_dict.update(current_result)

    with open("final_output_result.json", "w") as json_file:
        json.dump(results_dict, json_file)


def main():
    parser = argparse.ArgumentParser(description="Process videos and polygons file.")
    parser.add_argument(
        "video_list",
        help="Path to the video file or folder containing video files.",
    )
    parser.add_argument(
        "polygons_path",
        help="Path to the polygons file.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.video_list):
        video_list = [
            os.path.join(args.video_list, video)
            for video in os.listdir(args.video_list)
            if video.endswith(".mp4")
        ]
    elif os.path.isfile(args.video_list) and args.video_list.endswith(".mp4"):
        video_list = [args.video_list]
    else:
        print("Invalid input. Please provide a valid video file or folder for input 1.")
        return

    process_videos(video_list, args.polygons_path)


if __name__ == "__main__":
    main()
