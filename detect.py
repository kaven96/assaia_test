import cv2
import numpy as np
import json
import os
import argparse


class VehicleDetector:
    def __init__(
        self,
        video_name,
        cascade_path,
        polygons_path,
        c_scaling=1.22,
        c_neighbors=7,
        write_frames=False,
    ):
        self.video_name = video_name
        self.cascade_path = cascade_path
        self.polygons_path = polygons_path
        self.c_scaling = c_scaling
        self.c_neighbors = c_neighbors
        self.write_frames = write_frames
        self.cap = None
        self.car_cascade = None
        self.polygons = None
        self.frame_number = 0
        self.detected_vehicles = {}
        self.time_ranges = {}
        self.pts_int = None

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

    def detect_vehicles(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_number += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            dilated = cv2.dilate(blur, np.ones((3, 3)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            cars = self.car_cascade.detectMultiScale(
                closing, self.c_scaling, self.c_neighbors
            )

            for x, y, w, h in cars:
                x, y, w, h = int(x), int(y), int(w), int(h)
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
                    elif (
                        self.frame_number - 1
                        > list(self.time_ranges.values())[-1]["end"]
                    ):
                        t_n = max(self.time_ranges.keys()) + 1
                        self.time_ranges[t_n] = {
                            "start": self.frame_number,
                            "end": self.frame_number,
                        }
                    elif (
                        self.frame_number - 1
                        == list(self.time_ranges.values())[-1]["end"]
                    ):
                        t_n = max(self.time_ranges.keys())
                        self.time_ranges[t_n]["end"] = self.frame_number

            cv2.polylines(
                frame, [self.pts_int], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # cv2.imshow('Output', frame)

            if self.write_frames:
                output_folder = "frames"
                frame_filename = f"{output_folder}/frame_{self.frame_number}.jpg"
                cv2.imwrite(frame_filename, frame)

            if cv2.waitKey(1) == 27:
                break

        self.export_time_ranges()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection Script")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("polygon_file", help="Path to the polygon file")
    # parser.add_argument('output_json', help='Filename for output JSON')

    args = parser.parse_args()

    video_name = args.video_file
    cascade_path = "cars.xml"
    polygons_path = args.polygon_file

    # Initialize and run the vehicle detector
    detector = VehicleDetector(
        video_name, cascade_path, polygons_path, write_frames=False
    )
    detector.load_video()
    detector.load_cascade()
    detector.load_polygons()
    detector.detect_vehicles()


if __name__ == "__main__":
    main()
