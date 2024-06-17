import numpy as np
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

source = "video.mp4"  
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error al leer el archivo" 
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define areas of interest (adjust coordinates as needed)
area1 = [(110, 330), (400, 330), (400, 390), (110, 390)]
area2 = [(670, 330), (980, 330), (980, 390), (670, 390)]

# Initialize video writer
video_writer = cv2.VideoWriter("contador_de_objetos.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, nn_budget=None)

classes_to_count = [2, 3, 4, 5, 7]  
counter = [0, 0]  # Counters for each area
direction_counts = {"Norte": 0, "Sur": 0, "Este": 0, "Oeste": 0}  

margin_x = 50
margin_y = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('El Video esta vacio')
        break

    # Perform object detection and tracking
    results = model.track(frame, persist=True, show=False, classes=classes_to_count, tracker='bytetrack.yaml')

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu()
        confidences = result.boxes.conf.cpu()
        class_ids = result.boxes.cls.cpu()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    areas = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        bbox = track.to_tlbr()
        class_id = track.det_class
        
        # Draw bounding box and track ID
        x1, y1, x2, y2 = map(int, bbox)
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track.track_id), (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check direction and count if entering an area
        if len(track.mean) >= 2:
            prev_centroid = track.mean[-2] if isinstance(track.mean[-2], (list, tuple, np.ndarray)) else (0, 0)  # Default to (0, 0) if not a sequence
            prev_centroid_x, prev_centroid_y = prev_centroid
            if cv2.pointPolygonTest(np.array(area1, np.int32), (centroid_x, centroid_y), False) >= 0 and cv2.pointPolygonTest(np.array(area1, np.int32), (prev_centroid_x, prev_centroid_y), False) < 0:
                counter[0] += 1
                if centroid_x > prev_centroid_x:
                    direction_counts["Norte"] += 1
                else:
                    direction_counts["Sur"] += 1
            elif cv2.pointPolygonTest(np.array(area2, np.int32), (centroid_x, centroid_y), False) >= 0 and cv2.pointPolygonTest(np.array(area2, np.int32), (prev_centroid_x, prev_centroid_y), False) < 0:
                counter[1] += 1
                if centroid_y > prev_centroid_y:
                    direction_counts["Este"] += 1
                else:
                    direction_counts["Oeste"] += 1

    # Display counts and areas
    cv2.putText(frame, f'Area1: {counter[0]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Area2: {counter[1]}', (670, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    # Display directions
    for i, (direction, count) in enumerate(direction_counts.items()):
        cv2.putText(frame, f'{direction}: {count}', (10, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to video
    #video_writer.write(frame)
    cv2.imshow("YOLOv8 Direction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
