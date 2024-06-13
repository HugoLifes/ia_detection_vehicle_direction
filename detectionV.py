import numpy as np
import cv2
import os
from ultralytics import YOLO,solutions
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
try:
    import torch
    GPU = torch.cuda.is_available() and not os.environ.get('USE_CPU')
    
except ModuleNotFoundError:
    GPU=False
    
    
def vehicle_detection():
    # modelo yolov8
    
    model = YOLO('yolov8n.pt') # modelo pre entrenado RT-DETR
    #model.fuse() # fusion de capas mejor rendimiento
    source = "video.mp4" # direccion del video
    cap = cv2.VideoCapture(source) #
    assert cap.isOpened(), "Error al leer el archivo" 
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    line_points = [(20, 400), (1080, 400)]  # line o puntos de la zona
    classes_to_count = [2,3,4,5,7] # objetos a contar 
    
    video_writer = cv2.VideoWriter("contador_de_objetos.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps,(w,h))
    
    counter = solutions.ObjectCounter(view_img=True,reg_pts=line_points, classes_names=model.names,draw_tracks=True, line_thickness=2)
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print('El Video esta vacio')
            break
        
        tracks = model.track(frame, persist=True, show=False, classes=classes_to_count, tracker='bytetrack.yaml')
        predict = model.predict(frame)
        print(predict)
        frame = counter.start_counting(frame,tracks)
        
        video_writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()



def vehicle_detection_directions():
    # modelo yolov8
    
    model = YOLO('yolov8n.pt') # modelo pre entrenado RT-DETR
    #model.fuse() # fusion de capas mejor rendimiento
    source = "video.mp4" # direccion del video
    cap = cv2.VideoCapture(source) #
    assert cap.isOpened(), "Error al leer el archivo" 
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    line_points = [(20, 400), (1080, 400)]  # line o puntos de la zona
    classes_to_count = [2,3,4,5,7] # objetos a contar 
    
    video_writer = cv2.VideoWriter("contador_de_objetos.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps,(w,h))
    
    counter = solutions.ObjectCounter(view_img=True,reg_pts=line_points, classes_names=model.names,draw_tracks=True, line_thickness=2)
    
    # Dictionaries to track counts for each direction
    direction_counts = {
        "Norte": 0,
        "Sur": 0,
        "Este": 0,
        "Oeste": 0
    }
    
    # Dictionaries to track the previous positions of each vehicle
    previous_positions = defaultdict(lambda: None)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print('El Video esta vacio')
            break
        
        results = model.track(frame, persist=True, show=False, classes=classes_to_count, tracker='bytetrack.yaml')
        boxes = results[0].boxes.xyxy.cpu()  # Get the boxes in xyxy format
        track_ids = results[0].boxes.id.int().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Get previous position for this track
            prev_pos = previous_positions[track_id]

            # If this is not the first frame for this track
            if prev_pos is not None:
                prev_x, prev_y = prev_pos

                # Determine direction based on movement
                if abs(center_x - prev_x) > abs(center_y - prev_y):
                    direction = "Este" if center_x > prev_x else "Oeste"
                else:
                    direction = "Sur" if center_y > prev_y else "Norte"
               
                direction_counts[direction] += 1  # Increment direction count
                
                cv2.putText(frame, direction, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store current position for next frame
            previous_positions[track_id] = (center_x, center_y)

        # Display direction counts
        for i, (direction, count) in enumerate(direction_counts.items()):
            cv2.putText(frame, f"{direction}: {count}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame = counter.start_counting(frame, results)  # Show counter and line for reference
        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


