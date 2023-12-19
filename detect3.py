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
        track_boxes = {}

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

            track_boxes[self.frame_number] = []

            # if self.frame_number - 1 in track_boxes:
            #     print(self.frame_number)
            #     for t_box in track_boxes[self.frame_number - 1]:
            #         for box in contours:
            #             if cv2.contourArea(box) > 2000:
            #                 x, y, w, h = cv2.boundingRect(box)
            #                 if (
            #                     np.abs(t_box[0] - x) <= 200
            #                     or np.abs(t_box[1] - y) <= 200
            #                 ):
            #                     continue
            #                 else:
            #                     t_n = max(self.time_ranges.keys()) + 1
            #                     self.time_ranges[t_n] = {
            #                         "start": self.frame_number,
            #                         "end": self.frame_number,
            #                     }
            #                     # print(self.frame_number)
            #                     old_coord_list = track_boxes[self.frame_number - 1]
            #                     new_coord_list = old_coord_list.append((x,y,w,h))
            #                     track_boxes[self.frame_number] = new_coord_list
            #             else:
            #                 old_coord_list = track_boxes[self.frame_number - 1]
            #                 print(old_coord_list)
            #                 update_dict(
            #                     track_boxes,
            #                     self.frame_number - 1,
            #                     old_coord_list,
            #                     self.frame_number,
            #                     old_coord_list
            #                 )
            #                 # self.frame_number += 1

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

                    if any(
                        cv2.pointPolygonTest(self.pts_int, corner, False) >= 0
                        for corner in rect_corners
                    ) or any(
                        cv2.pointPolygonTest(self.pts_int, edge, False) >= 0
                        for edge in rect_edges
                    ):
                        if not self.time_ranges:
                            self.time_ranges[0] = {
                                "start": self.frame_number,
                                "end": self.frame_number,
                            }

                        for t_box in track_boxes[self.frame_number-1]:
                            # print('t_box:', t_box, 'box', box)
                            t_n = max(self.time_ranges.keys())
                            if calculate_iou(t_box, (x,y,w,h)) >= 0.5:
                                self.time_ranges[t_n]['end'] = self.frame_number
                            elif self.time_ranges[t_n]['end'] == self.frame_number-1:
                                    self.time_ranges[t_n]['end'] = self.frame_number
                            else:
                                self.time_ranges[t_n+1] = {'start': self.frame_number, 'end': self.frame_number}
                        
                        track_boxes[self.frame_number].append((x, y, w, h))

                        
                            # print("TYPE", type(self.frame_number))
                        #     track_boxes[self.frame_number] = [(x, y, w, h)]
                        # elif (
                        #     self.frame_number - 1
                        #     > list(self.time_ranges.values())[-1]["end"]
                        # ):
                        #     t_n = max(self.time_ranges.keys()) + 1
                        #     self.time_ranges[t_n] = {
                        #         "start": self.frame_number,
                        #         "end": self.frame_number,
                        #     }
                        #     track_boxes[self.frame_number] = [(x, y, w, h)]
                        # elif (
                        #     self.frame_number - 1
                        #     == list(self.time_ranges.values())[-1]["end"]
                        # ):
                        #     t_n = max(self.time_ranges.keys())
                        #     self.time_ranges[t_n]["end"] = self.frame_number
                        #     for idx in range(
                        #         0, len(track_boxes[self.frame_number - 1])
                        #     ):
                        #         if (
                        #             np.abs(
                        #                 x - track_boxes[self.frame_number - 1][idx][0]
                        #             )
                        #             <= 200
                        #             or np.abs(
                        #                 y - track_boxes[self.frame_number - 1][idx][1]
                        #             )
                        #             <= 200
                        #         ):
                        #             update_dict(
                        #                 track_boxes,
                        #                 self.frame_number - 1,
                        #                 track_boxes[self.frame_number - 1][idx],
                        #                 self.frame_number,
                        #                 (x, y, w, h),
                        #             )
                        #         else:
                        #             old_coord_list = track_boxes[self.frame_number - 1]
                        #             new_coord_list = old_coord_list.append((x, y, w, h))
                        #             del track_boxes[self.frame_number - 1]
                        #             track_boxes[self.frame_number] = new_coord_list

            cv2.polylines(
                frame, [self.pts_int], isClosed=True, color=(0, 255, 0), thickness=2
            )

            cv2.imshow("Output", frame)

            # if self.write_frames:
            output_folder = "frames"
            frame_filename = f"{output_folder}/frame_{self.frame_number}.jpg"
            cv2.imwrite(frame_filename, frame)

            # print(self.frame_number)
            self.frame_number += 1

            if cv2.waitKey(1) == 27:
                break

        self.export_time_ranges()
        self.cap.release()
        cv2.destroyAllWindows()


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
