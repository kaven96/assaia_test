from sklearn.model_selection import train_test_split
from detect import VehicleDetector
from evaluate import evaluate_predictions
import json
import os
import numpy as np

class DetectorWrapper:
    def __init__(self, cascade_path, polygons_path, write_frames=False, c_scaling=1.1, c_neighbors=10):
        self.cascade_path = cascade_path
        self.polygons_path = polygons_path
        self.write_frames = write_frames
        self.c_scaling = c_scaling
        self.c_neighbors = c_neighbors

    def fit(self, X, y=None):
        # This is where you can perform your training logic
        return self

    def predict(self, X):
        results_dict = {}
        for video_name in X:
            print(f"Processing video: {video_name}")
            detector = VehicleDetector(video_name, self.cascade_path, self.polygons_path,
                                       c_scaling=self.c_scaling, c_neighbors=self.c_neighbors,
                                       write_frames=self.write_frames)
            detector.load_video()
            detector.load_cascade()
            detector.load_polygons()
            detector.detect_vehicles()

            with open('output_result.json', 'r') as json_file:
                current_result = json.load(json_file)

            results_dict.update(current_result)

        return results_dict

def load_ground_truth(ground_truth_file):
    with open(ground_truth_file, 'r') as f:
        return json.load(f)

def tune_hyperparameters(video_list, ground_truth_file):
    ground_truth = load_ground_truth(ground_truth_file)
    train_videos, test_videos = train_test_split(video_list, test_size=0.2, random_state=42)

    best_params = {'c_neighbors': None, 'c_scaling': None}
    best_score = 0.0  # or -float('inf') for minimization tasks

    for c_scaling in np.linspace(1.01, 2.01, 20):
        for c_neighbors in np.arange(1, 15, 2):
            detector_wrapper = DetectorWrapper('cars.xml', 'polygons.json',
                                              c_scaling=c_scaling, c_neighbors=c_neighbors)
            detector_wrapper.fit(train_videos)
            predictions = detector_wrapper.predict(test_videos)

            with open('final_output_result.json', 'w') as json_file:
                json.dump(predictions, json_file)

            precision, recall, accuracy, f1_score = evaluate_predictions(ground_truth_file, 'final_output_result.json')

            # Update best hyperparameters if a better score is achieved
            if f1_score > best_score:
                best_score = f1_score
                best_params['c_neighbors'] = c_neighbors
                best_params['c_scaling'] = c_scaling

    print("Best Hyperparameters:", best_params)
    print("Best F1 Score:", best_score)

if __name__ == "__main__":
    video_folder = 'videos'
    video_list = [f"{video_folder}/{video}" for video in os.listdir(video_folder) if video.endswith('.mp4')]
    ground_truth_file = 'time_intervals.json'

    tune_hyperparameters(video_list, ground_truth_file)
