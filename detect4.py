import cv2
import numpy as np
import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFilter
import torch
import numpy as np


class VehicleDetector:
    def __init__(
        self,
        video_name,
        polygons_path,
        write_frames=True,
    ):
        self.video_name = video_name
        self.polygons_path = polygons_path
        self.write_frames = write_frames
        self.cap = None
        self.car_cascade = None
        self.polygons = None
        self.frame_number = 0
        self.detected_vehicles = {}
        self.time_ranges = {}
        self.pts_int = None
        self.backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
        self.kernel = np.ones((3, 3), np.uint8)
        self.kernel2 = None
        self.tracker = cv2.TrackerKCF_create()

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_name)

    def load_cascade(self):
        self.car_cascade = cv2.CascadeClassifier(self.cascade_path)

    def load_polygons(self):
        video_name = os.path.basename(self.video_name)
        with open(self.polygons_path, "r") as json_file:
            self.polygons = json.load(json_file)

        pts = np.array(self.polygons[video_name], np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.pts_int = pts.astype(np.int32)

    def export_time_ranges(self):
        video_name = os.path.basename(self.video_name)
        result_dict = {}
        if video_name not in result_dict:
            result_dict[video_name] = []

        for _, interval in self.time_ranges.items():
            result_dict[video_name].append([interval["start"], interval["end"]])

        with open("output_result.json", "w") as json_file:
            json.dump(result_dict, json_file)

    def xyxy_to_xywh(self, xyxy):
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)

        return x_center, y_center, width, height

    def detect_vehicles(self):
        inside_polygon = []
        trackers = []  # Keep track of trackers for each object

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fgmask = self.backgroundObject.apply(rgb_frame)
            _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
            fgmask = cv2.erode(fgmask, self.kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, self.kernel2, iterations=6)

            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for box in contours:
                if cv2.contourArea(box) > 2000:
                    x, y, w, h = cv2.boundingRect(box)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    rect_corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                    rect_edges = [
                        (int((x + x + w) // 2), y),
                        (x + w, int((y + y + h) // 2)),
                        (int((x + x + w) // 2), y + h),
                        (x, int((y + y + h) // 2)),
                    ]

                    inside = any(
                        cv2.pointPolygonTest(self.pts_int, corner, False) >= 0
                        for corner in rect_corners
                    ) or any(
                        cv2.pointPolygonTest(self.pts_int, edge, False) >= 0
                        for edge in rect_edges
                    )

                    if inside:
                        if (x, y, w, h) not in inside_polygon:
                            # Mark the object as inside and record the start frame
                            inside_polygon.append((x, y, w, h, self.frame_number))
                            if not self.time_ranges:
                                self.time_ranges[self.frame_number] = {
                                    "start": self.frame_number,
                                    "end": self.frame_number,
                                }

                            # Initialize tracker for the object
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, (x, y, w, h))
                            trackers.append(tracker)
                    else:
                        if (x, y, w, h) in inside_polygon:
                            # Update the time_ranges dictionary and remove the object from inside_polygon
                            obj_index = inside_polygon.index((x, y, w, h, self.frame_number))
                            start_frame = inside_polygon[obj_index][-1]
                            self.time_ranges[start_frame]["end"] = self.frame_number
                            inside_polygon.pop(obj_index)
                            trackers.pop(obj_index)

            for obj_index, tracker in enumerate(trackers):
                # Update the tracker for each object
                success, new_box = tracker.update(frame)
                if success:
                    # Get the new bounding box coordinates
                    x, y, w, h = map(int, new_box)

                    # Draw the updated bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.polylines(
                frame, [self.pts_int], isClosed=True, color=(0, 255, 0), thickness=2
            )

            cv2.imshow("Output", frame)

            # if self.write_frames:
            output_folder = "frames"
            frame_filename = f"{output_folder}/frame_{self.frame_number}.jpg"
            cv2.imwrite(frame_filename, frame)

            # Check for leaving events
            for obj_index, (x, y, w, h, start_frame) in enumerate(inside_polygon):
                if obj_index < len(trackers) and trackers[obj_index] is None:
                    # The object has left the polygon
                    self.time_ranges[start_frame]["end"] = self.frame_number

            # print(self.frame_number)
            self.frame_number += 1

            if cv2.waitKey(1) == 27:
                break

        self.export_time_ranges()
        self.cap.release()
        cv2.destroyAllWindows()

def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def update_dict(dictionary, old_frame, old_coordinates, new_frame, new_coordinates):
    if old_frame in dictionary:
        # If the old_frame is present in the dictionary, find and replace the old_coordinates with new_coordinates
        coordinates_list = dictionary[old_frame]
        for i, coordinates in enumerate(coordinates_list):
            if coordinates == old_coordinates:
                coordinates_list[i] = new_coordinates
                break  # Stop searching once the first occurrence is replaced

        # Check if the new_frame is different from the old_frame
        if old_frame != new_frame:
            # Remove the old_frame entry and add the new_frame entry
            dictionary[new_frame] = dictionary.pop(old_frame)
    else:
        # If the old_frame is not present, add a new entry to the dictionary with new_frame and new_coordinates
        dictionary[new_frame] = [new_coordinates]

def calculate_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Check for non-overlapping rectangles
    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0

    # Calculate the areas of rectangles and the intersection
    area_rect1 = w1 * h1
    area_rect2 = w2 * h2
    area_intersection = w_intersection * h_intersection

    # Calculate the IOU
    iou = area_intersection / (area_rect1 + area_rect2 - area_intersection)
    return iou

def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection Script")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("polygon_file", help="Path to the polygon file")
    # parser.add_argument('output_json', help='Filename for output JSON')

    args = parser.parse_args()

    video_name = args.video_file
    polygons_path = args.polygon_file

    # Initialize and run the vehicle detector
    detector = VehicleDetector(video_name, polygons_path, write_frames=True)
    detector.load_video()
    detector.load_polygons()
    detector.detect_vehicles()


if __name__ == "__main__":
    main()
