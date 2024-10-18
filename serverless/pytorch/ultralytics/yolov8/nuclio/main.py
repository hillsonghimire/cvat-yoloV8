import json
import base64
from PIL import Image
import io
import torch
import numpy as np
from ultralytics import YOLO

def to_cvat_rectangle(box: list):
    xtl, ytl, xbr, ybr = box
    return [xtl, ytl, xbr, ybr]

def init_context(context):
    context.logger.info("Init context...  0%")
    model_path = "/opt/nuclio/model.pt"
    model = YOLO(model_path, task="detect")
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.15))
    context.user_data.model.conf = threshold
    image = Image.open(buf)

    # Run detection model
    yolo_results = context.user_data.model(image, conf=threshold)[0]
    labels = yolo_results.names

    results = []
    if len(yolo_results.boxes) > 0:
        for box in yolo_results.boxes:
            xyxy = box.xyxy.numpy()[0]  # Bounding box coordinates
            confidence = box.conf.item()  # Confidence score
            class_id = box.cls.item()  # Class ID
            xtl, ytl, xbr, ybr = map(int, xyxy)
            label = labels.get(class_id, "unknown")
            cvat_rectangle = to_cvat_rectangle([xtl, ytl, xbr, ybr])
            results.append({
                "confidence": str(confidence),
                "label": label,
                "type": "rectangle",  # Change type to "rectangle"
                "points": cvat_rectangle,
                "attributes": []
            })

    return context.Response(body=json.dumps(results), headers={},
                            content_type='application/json', status_code=200)
