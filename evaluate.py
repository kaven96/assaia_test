import sys
import json
import numpy as np
import os


def calculate_metrics(ground_truth, predictions):
    tp, fp, fn = 0, 0, 0

    for video_name, gt_intervals in ground_truth.items():
        pred_intervals = predictions.get(video_name, [])

        for pred_interval in pred_intervals:
            found_match = False

            for gt_interval in gt_intervals:
                if (
                    pred_interval[0] <= gt_interval[1]
                    and pred_interval[1] >= gt_interval[0]
                ):
                    tp += 1
                    found_match = True
                    break

            if not found_match:
                fp += 1

        fn += len(
            [
                1
                for gt_interval in gt_intervals
                if not any(
                    pred_interval[0] <= gt_interval[1]
                    and pred_interval[1] >= gt_interval[0]
                    for pred_interval in pred_intervals
                )
            ]
        )

    return tp, fp, fn


def calculate_precision_recall_accuracy_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, accuracy, f1


def evaluate_predictions(ground_truth_file, predictions_file):
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    tp, fp, fn = calculate_metrics(ground_truth, predictions)
    precision, recall, accuracy, f1 = calculate_precision_recall_accuracy_f1(tp, fp, fn)

    if not np.isfinite(precision):
        print("Precision is not finite. Set to 0.")
        precision = 0

    if not np.isfinite(recall):
        print("Recall is not finite. Set to 0.")
        recall = 0

    if not np.isfinite(accuracy):
        print("Accuracy is not finite. Set to 0.")
        accuracy = 0

    if not np.isfinite(f1):
        print("F1 Score is not finite. Set to 0.")
        f1 = 0

    return precision, recall, accuracy, f1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python evaluate_intervals.py <ground_truth_file> <predictions_file>"
        )
        sys.exit(1)

    ground_truth_file = sys.argv[1]
    predictions_file = sys.argv[2]
    video_name = sys.argv[3]
    video_name = os.path.basename(video_name)

    precision, recall, accuracy, f1 = evaluate_predictions(
        ground_truth_file, predictions_file
    )

    with open(f"metrics_{video_name}.txt", "w") as f:
        f.write(f"Precision: {precision}")
        f.write("\n")
        f.write(f"Recall: {recall}")
        f.write("\n")
        f.write(f"Accuracy: {accuracy}")
        f.write("\n")
        f.write(f"F1 Score: {f1}")
        f.write("\n")
