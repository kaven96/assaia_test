import cv2

# import clip
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class ImageSegmentationModel:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def calculate_relative_center(self, box, original_size):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        rel_center_x = center_x / original_size[0]
        rel_center_y = center_y / original_size[1]

        return {"x": rel_center_x, "y": rel_center_y}

    def draw_box(self, image, box_coordinates, outline_color="red", line_width=2):
        draw = ImageDraw.Draw(image)

        # Draw the box
        draw.rectangle(box_coordinates, outline=outline_color, width=line_width)

        return image

    def process_image(self, image_path, text_prompt):
        image = self.load_image(image_path)
        # blurred_image = image.filter(ImageFilter.GaussianBlur(2))
        blurred_image = image

        inputs = self.processor(
            text=[text_prompt], images=blurred_image, return_tensors="pt"
        )

        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        i = 0
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.01
        )

        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        max_score_index = torch.argmax(scores)
        max_score_box = [round(i, 2) for i in boxes[max_score_index].tolist()]        

        rel_center = self.calculate_relative_center(max_score_box, image.size)

        result_image = self.draw_box(image, max_score_box)
        # print(boxes[max_score_index])
        segmentation_info = {'rel_center': rel_center}

        return result_image, [segmentation_info]
