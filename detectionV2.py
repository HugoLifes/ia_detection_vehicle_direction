from matplotlib.pyplot import annotate
import numpy as np
import cv2
import os
from ultralytics import YOLO,solutions


try:
    import torch
    GPU = torch.cuda.is_available() and not os.environ.get('USE_CPU')

except ModuleNotFoundError:
    GPU = False

# modelo yolov8
# Variables globales para la interfaz (sin cambios)
drawing = False
ix, iy = -1, -1
current_shape = "rectangle"
shapes = []
current_frame = None


# Function to handle mouse events
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, current_shape, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if current_shape == "rectangle":
                cv2.rectangle(current_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            elif current_shape == "line":
                cv2.line(current_frame, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        shapes.append({
            "type": current_shape,
            "coords": [(ix, iy), (x, y)]
        })
        print(f"{current_shape.capitalize()} added: {shapes[-1]}")


# Función para verificar si un punto está dentro de un rectángulo o línea
def is_inside_shape(point, shape):
    (x, y) = point
    (x1, y1), (x2, y2) = shape["coords"]

    if shape["type"] == "rectangle":
        return x1 <= x <= x2 and y1 <= y <= y2
    elif shape["type"] == "line":
        # Calcula la distancia del punto a la línea
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance_to_line = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_length
        return distance_to_line <= 5  # Consideramos 5 píxeles de tolerancia


model = YOLO("yolov8n.pt")  # modelo pre entrenado RT-DETR
# model.fuse() # fusion de capas mejor rendimiento
source = "video.mp4"  # direccion del video
cap = cv2.VideoCapture(source)  #
assert cap.isOpened(), "Error al leer el archivo"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

line_points = [(0, 0), (0, 0)]  # line o puntos de la zona
classes_to_count = [2, 3, 4, 5, 7]  # objetos a contar

video_writer = cv2.VideoWriter(
    "contador_de_objetos.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=None,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)
counted_objects = set()

# Previous positions for direction detection
prev_positions = {}
# Set up the window and mouse callback function
cv2.namedWindow("Draw time in picture time in yolo trackin")
cv2.setMouseCallback("Draw time in picture time in yolo trackin", draw_shape)

# Dictionary to store object counts for each shape
counts = {i: 0 for i in range( len(shapes))}
prev_x, prev_y = None, None
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("El Video esta vacio")
        break
    
    current_frame = frame.copy()

    results = model.track(
        frame,
        persist=True,
        classes=classes_to_count,
        tracker="bytetrack.yaml",
        )
    #frame = results[0].plot()
 
    for result in results:
        boxes = result.boxes.xyxy.cpu()[:4]
        track_id = result.boxes.id.int().tolist()
        
        for box, track_id in zip(boxes,track_id):
            x1, y1, x2, y2 = box # cuadrantes
            x_center = (x1 + x2) / 2  # posicion en x
            y_center = (y1 + y2) / 2  # posicion en y
            
            
            # Direction detection (simplified)
            prev_pos = prev_positions.get(track_id)
            if prev_pos:
                prev_x, prev_y = prev_pos
                if abs(x_center - prev_x) > abs(y_center - prev_y):
                    direction = "Derecha" if x_center > prev_x else "Izquierda"
                else:
                    direction = "Abajo" if y_center > prev_y else "Arriba"
            else:
                direction = "Detenido"  

            prev_positions[track_id] = (x_center, y_center)    
            cv2.putText(frame, direction, (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if the object is in any of the shapes AND hasn't been counted before
            for i, shape in enumerate(shapes):
                if is_inside_shape((x_center, y_center), shape) and track_id not in counted_objects:
                    counts[i] += 1
                    counted_objects.add(track_id) # Mark the object as counted  # Incrementa el contador de la forma correspondiente
    
    
        # Draw the shapes and display the counts on the frame
        for i, shape in enumerate(shapes):
            (x1, y1), (x2, y2) = shape["coords"]
            if shape["type"] == "rectangle":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                count_position = (x1, y1 - 10)
            elif shape["type"] == "line":
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)       # Red line
                count_position = ((x1 + x2) // 2, (y1 + y2) // 2)         # Center of the line

            count = counts.get(i, 0)  # Default to 0 if count doesn't exist
            cv2.putText(frame, f"Shape {i + 1}: {count}", count_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Green text
                    
    
    frame = counter.start_counting(frame,result) # Show counter and line for reference
    
    #cv2.imshow("Aforo Mode", frame)
    video_writer.write(frame)
     # Handle key presses for changing shape mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Switch to rectangle mode
        current_shape = "rectangle"
        print("Rectangle mode activated")
    elif key == ord('l'):  # Switch to line mode
        current_shape = "line"
        print("Line mode activated")
    elif key == ord('q'):  # Exit the loop
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
# Print the final counts to the console
for i, count in counts.items():
    print(f"Shape {i}: {count} objects")