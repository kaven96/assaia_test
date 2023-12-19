import os
from detect5 import VehicleDetector
import json


def process_videos(video_list):
    results_dict = {}

    for video_name in video_list:
        print(f"Processing video: {video_name}")
        detector = VehicleDetector(
            video_name,
            "polygons.json",
            write_frames=False,
        )
        detector.load_video()
        detector.load_polygons()
        detector.detect_vehicles()

        with open("output_result.json", "r") as json_file:
            current_result = json.load(json_file)

        results_dict.update(current_result)

    with open("final_output_result.json", "w") as json_file:
        json.dump(results_dict, json_file)


if __name__ == "__main__":
    video_folder = "videos"  
    video_list = [
        f"{video_folder}/{video}"
        for video in os.listdir(video_folder)
        if video.endswith(".mp4")
    ]
    process_videos(video_list)
