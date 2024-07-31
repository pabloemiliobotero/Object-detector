import cv2
from ultralytics import YOLO
import random
import os
PATH_TO_LABELS = os.path.join('.', 'model_data', 'coco_classes.txt')

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
model = YOLO("yolov8n.pt")
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[class_id % len(colors)]), 3)
                object_name = labels[int(class_id)]
                label = '%s: %d%%' % (object_name, int(score * 100))
                cv2.putText(frame, str(label), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)

    cv2.namedWindow("Object detector", cv2.WINDOW_NORMAL)
    ##cv2.setWindowProperty("Object detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Object detector', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Object detector', frame)
    ret, frame = cap.read()
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()