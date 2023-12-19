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
        write_frames=False,
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

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_name)

    def load_polygons(self):
        """
        Read polygon coordinates and store as pts_int
        """
        video_name = os.path.basename(self.video_name)
        with open(self.polygons_path, "r") as json_file:
            self.polygons = json.load(json_file)

        pts = np.array(self.polygons[video_name], np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.pts_int = pts.astype(np.int32)

    def export_time_ranges(self):
        """
        Export time_ranges dict to json
        """
        video_name = os.path.basename(self.video_name)
        result_dict = {}
        if video_name not in result_dict:
            result_dict[video_name] = []

        for _, interval in self.time_ranges.items():
            result_dict[video_name].append([interval["start"], interval["end"]])

        merged_intervals = merge_intervals(
            result_dict[video_name]
        )  # glue intervals like [18,20], [20, 35] together

        result_dict[video_name] = merged_intervals

        with open(
            "output_result.json", "w"
        ) as json_file:  # output_results.json name is hardcoded, workaround is in bash script
            json.dump(result_dict, json_file)

    def xyxy_to_xywh(self, xyxy):
        """
        Convert bounding box with coordinates (x1, y1, x2, y2) to bbox (x, y, w, h)
        Where x1 --- left vertical border coordinate
              x2 --- right vertical border coordinate
              y1 --- top horizontal border coordinate
              y2 --- bottom horizpntal border coordinate

              x, y --- top left corner of rectangle coordinates
              w --- width of the rectangle
              h --- height of the rectangle
        """
        x_min, y_min, x_max, y_max = xyxy
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)

        return x_center, y_center, width, height

    def detect_vehicles(self):
        """
        The main function for moving objects detection.

        The main idea here is to use Background Substraction method.

        It works as following:

        For each frame of the video:
        1. Calculate background and foreground mask using 2 last frames
        2. Apply erosion and dilation operations on foreground mask. Erosion suppresses thin lines,
           dilation makes lines that left thicker.
        3. Find all closed contours on the image
        4. For each contour with bbox with area > 2000 pixels (hyperparameter depending on video
           resolution and distance from camera to object):
                a. check, if object is inside polygon
                b. check if object was in polygon earlier, if yes --- move end_frame to the current frame
                   if no, add current frame_number to time_ranges dict as start_frame
                c. add current object to temporary list track_boxes
        """
        track_boxes = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

            fgmask = self.backgroundObject.apply(rgb_frame)  # extract foreground mask

            # apply morphological operations on fgmask:
            _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
            fgmask = cv2.erode(fgmask, self.kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, self.kernel2, iterations=6)

            # detect contours on the foreground mask
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

                    # check if object bbox is inside polygon
                    if any(
                        cv2.pointPolygonTest(self.pts_int, corner, False) >= 0
                        for corner in rect_corners
                    ) or any(
                        cv2.pointPolygonTest(self.pts_int, edge, False) >= 0
                        for edge in rect_edges
                    ):
                        # if time_ranges is empty, add first range
                        if not self.time_ranges:
                            self.time_ranges[0] = {
                                "start": self.frame_number,
                                "end": self.frame_number,
                            }
                            track_boxes.append((x, y, w, h))

                        t_n = max(
                            self.time_ranges.keys()
                        )  # get number of the latest range in time_ranges

                        # check if there were any objects inside the polygon:
                        if self.time_ranges[t_n]["end"] == self.frame_number - 1:
                            self.time_ranges[t_n]["end"] = self.frame_number
                        else:
                            # check if object stopped inside polygon and disappeared
                            #   from further foreground masks
                            for t_box in track_boxes:
                                # IOU threshold is 0.5 to skip inaccuracies of bbox
                                #   calculations
                                if calculate_iou(t_box, (x, y, w, h)) >= 0.5:
                                    self.time_ranges[t_n]["end"] = self.frame_number
                                    track_boxes.remove(t_box)
                                    track_boxes.append((x, y, w, h))
                                else:
                                    self.time_ranges[t_n + 1] = {
                                        "start": self.frame_number,
                                        "end": self.frame_number,
                                    }

            # draw area of interest above image. Needed for algorythm analysis
            cv2.polylines(
                frame, [self.pts_int], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # uncomment below to draw frames when running script:
            # cv2.imshow("Output", frame)

            # if write_frames is True, save frames with bboxes and polygon
            if self.write_frames:
                output_folder = "frames"
                frame_filename = f"{output_folder}/frame_{self.frame_number}.jpg"
                cv2.imwrite(frame_filename, frame)

            # increment frame number
            self.frame_number += 1

            # leave while loop at the end of the video
            if cv2.waitKey(1) == 27:
                break

        self.export_time_ranges()
        self.cap.release()
        cv2.destroyAllWindows()


def merge_intervals(intervals):
    '''
    Merge time intervals in list together, if beginning of one interval is equal
        to the end of another interval or equal to end + 1.
        For example [[18,20], [20,35]] -> [[18, 35]] 
    '''
    if not intervals:
        return []

    # Sort intervals based on the start value
    intervals.sort(key=lambda x: x[0])

    merged_intervals = [intervals[0]]

    for interval in intervals[1:]:
        current_start, current_end = merged_intervals[-1]
        start, end = interval

        # Merge intervals if they overlap or have a difference of 1
        if current_end >= start or current_end + 1 == start:
            merged_intervals[-1] = [current_start, max(current_end, end)]
        else:
            merged_intervals.append(interval)

    return merged_intervals


def update_dict(dictionary, old_frame, old_coordinates, new_frame, new_coordinates):
    '''
    Update dictionary key or value, or both
    '''
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
    '''
    Calculate Intersection over Union for two rectangles
    '''
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
    '''
    Main function parsing arguments and running functions in appropriate order
    '''
    parser = argparse.ArgumentParser(description="Vehicle Detection Script")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("polygon_file", help="Path to the polygon file")

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
