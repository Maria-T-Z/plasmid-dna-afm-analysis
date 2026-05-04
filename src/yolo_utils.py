import numpy as np
from ultralytics import YOLO


def load_yolo_model(weights_path: str) -> YOLO:
    """
    Loading trained YOLO model to get crops (predictions)
    """
    return YOLO(weights_path)


def get_bounding_boxes(model: YOLO, img_path: str, conf_threshold: float = 0.5) -> list:
    """
    Getting predictions
    Args:
        model (YOLO): The loaded YOLO model.
        img_path (str): Path to the 8-bit .png image for detection.
        conf_threshold (float, optional): Minimum confidence score for a detection. Defaults to 0.5.
    """
    results = model.predict(source=img_path, conf=conf_threshold, verbose=False)

    boxes_list = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            boxes_list.append([x1, y1, x2, y2])
    return boxes_list