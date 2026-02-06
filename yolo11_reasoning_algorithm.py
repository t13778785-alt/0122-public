import os
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from logger import Logger
log = Logger().logger

class CommonInferenceService:
    def __init__(self, args):
        self.args = args
        self.model = self.load_model()

    def load_data(self, data_path):
        image = cv2.imread(data_path)
        if image is None:
            raise ValueError(f"Failed to load image from {data_path}")
        return image

    def load_model(self):
        if os.path.isfile(self.args.model_path):
            model = YOLO(self.args.model_path)
        else:
            for file in os.listdir(self.args.model_path):
                if file.endswith(('.pt', '.onnx')):
                    model = YOLO(os.path.join(self.args.model_path, file))
                    break
        return model

    def inference(self, data):
        result = {"data_name": data['data_name']}
        log.info("===============> start load " + data['data_name'] + " <===============")

        image = self.load_data(data['data_path'])
        results = self.model(image)  # this returns a list of Results objects

        predictions = []
        for res in results:
            try:
                # Access the detection results directly from the 'boxes' attribute
                boxes = res.boxes.xyxy  # Bounding boxes in [xmin, ymin, xmax, ymax]
                confidences = res.boxes.conf  # Confidence scores
                class_ids = res.boxes.cls  # Class IDs
                class_names = res.names  # Class names

                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        xmin, ymin, xmax, ymax = boxes[i]
                        confidence = confidences[i]
                        cls = class_ids[i]
                        class_name = class_names[int(cls)] if class_names else f"Class {int(cls)}"
                        predictions.append({
                            "class": class_name,
                            "confidence": float(confidence),
                            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
            except Exception as e:
                log.error(f"Error processing result: {str(e)}")

        result['predictions'] = predictions
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='dubhe serving')
    parser.add_argument('--model_path', type=str, default='./yolo11n.pt',
                        help="path to model file or directory containing model file")
    parser.add_argument('--use_gpu', action='store_true',
                        help="use gpu for inference")
    args = parser.parse_args()

    server = CommonInferenceService(args)
    image_path = "./test.jpg"
    data = {"data_name": "test_image", "data_path": image_path}
    re = server.inference(data)
    print(re)
