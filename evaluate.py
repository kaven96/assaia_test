import json
import argparse


class TimeIntervalEvaluator:
    '''
    True positive is length of all intersections divided by length of all ground truth 
        intervals.
    False positive is length of parts of predicted ranges not intersecting ground truth 
        ranges divided by by length of all ground truth intervals.
    False negative is the length of parts of ground truth ranges that are not 
        intersecting with predicted ranges divided by length of all ground truth intervals
    '''
    def __init__(self, ground_truth, prediction):
        self.ground_truth = self.load_json(ground_truth)
        self.prediction = self.load_json(prediction)
        self.intersection_lengths = []
        self.false_positive_lengths = []
        self.false_negative_lengths = []

    def load_json(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    def calculate_metrics(self, video_name):
        if video_name not in self.prediction:
            raise ValueError(f"No predictions found for video: {video_name}")

        ground_truth_intervals = self.ground_truth.get(video_name, [])
        predicted_intervals = self.prediction[video_name]

        intersection_length = 0
        false_positive_length = 0
        false_negative_length = 0

        # calculate intersection of intervals in gt and predicted
        for predicted_interval in predicted_intervals:
            for gt_interval in ground_truth_intervals:
                intersection_start = max(predicted_interval[0], gt_interval[0])
                intersection_end = min(predicted_interval[1], gt_interval[1])

                if intersection_start < intersection_end:
                    intersection_length += intersection_end - intersection_start

        # calculate extra length of prediction
        false_positive_length = (
            sum(
                [
                    predicted_interval[1] - predicted_interval[0]
                    for predicted_interval in predicted_intervals
                ]
            )
            - intersection_length
        )

        # calculate extra length of ground truth
        false_negative_length = (
            sum(
                [
                    gt_interval[1] - gt_interval[0]
                    for gt_interval in ground_truth_intervals
                ]
            )
            - intersection_length
        )

        self.intersection_lengths.append(intersection_length)
        self.false_positive_lengths.append(false_positive_length)
        self.false_negative_lengths.append(false_negative_length)

    def calculate_all_metrics(self):
        '''
        TP, FP, FN are not interesting by themselves. 
        Their proportions are more interesting, so we will use popular metrics for it:
        - precision
        - recall
        - accuracy
        - f1
        '''
        for video_name in self.prediction.keys():
            self.calculate_metrics(video_name)

        total_intersection = sum(self.intersection_lengths)
        total_ground_truth = sum(
            [
                sum([gt_interval[1] - gt_interval[0] for gt_interval in intervals])
                for intervals in self.ground_truth.values()
            ]
        )

        tp = total_intersection / total_ground_truth
        fp = sum(self.false_positive_lengths) / total_ground_truth
        fn = sum(self.false_negative_lengths) / total_ground_truth

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        accuracy = total_intersection / total_ground_truth

        # F1 Score calculation
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        return precision, recall, accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate time intervals prediction.")
    parser.add_argument(
        "ground_truth_json_path", help="Path to the ground truth JSON file"
    )
    parser.add_argument("prediction_json_path", help="Path to the prediction JSON file")

    args = parser.parse_args()

    evaluator = TimeIntervalEvaluator(
        args.ground_truth_json_path, args.prediction_json_path
    )
    precision, recall, accuracy, f1 = evaluator.calculate_all_metrics()

    with open(f"metrics_all.txt", "w") as f:
        f.write(f"Precision: {precision}")
        f.write("\n")
        f.write(f"Recall: {recall}")
        f.write("\n")
        f.write(f"Accuracy: {accuracy}")
        f.write("\n")
        f.write(f"F1 Score: {f1}")
        f.write("\n")


if __name__ == "__main__":
    main()
