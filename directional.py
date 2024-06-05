import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.interpolate import CubicSpline
import deepSort_method as dpM
import interpolate_method as interpolate


CONFIDENCE_THRESHOLD = 0.5  # nivel de confianza
try:
    import torch

    GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU")
except ModuleNotFoundError:
    GPU = False


def id_detection_directional():
    # modelo yoloV8 la version mas estable y precisa

    model = YOLO("yolov8n.pt")  # carga del modelo

    model.fuse()  # mejoras las capas del modelo para un mejor rendimiento

    min_trajectory_length = 5  # establece el número mínimo de puntos en la trayectoria de un objeto que se deben considerar para calcular la curvatura y determinar si está girando
    # definir vehiculos y contadores
    vehicle_clases = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","unknown",]
    directions = ["arriba","abajo","izquierda","derecha","detenido","giro"]  # diccionario de direcciones a tomar en cuenta
    counts = {cls: {dir: 0 for dir in directions} for cls in vehicle_clases}
    cap = cv2.VideoCapture("video.mp4")  # captura video
    trajectories = {}  # diccionario para almacenar la trayectoria de los objetos
    #angulo por metodo deepsort
    #angle_threshold = 0.5  # umbral de distancia minima recorrida por pixel
    # Diccionario para almacenar la dirección de cada objeto
    
    #angulo por metodo interpolacion
    angle_threshold = 45
    max_age = 55  # numero maximo de frames que un objeto puede desaparecer antes de ser eliminado
    giro_threshold = 0.3 # giro por metodo de interpolacion
    epsilon = 1e-6
    tracker = DeepSort(max_age=max_age, embedder_gpu=GPU)

    # Diccionario para rastrear objetos contados (ahora por ID)
    counted_objects = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame is None:
            continue
        detections = []
        # deteccion de vehiculos
        results = model(frame)

        # funcion para realizar el metodo por deepSort (no tan eficiente sin modelo cnn)
        # cls = dpM.deepSortMethod(results, detections, tracker, frame,CONFIDENCE_THRESHOLD, vehicle_clases, trajectories, angle_threshold,min_trajectory_length,counted_objects,counts)
        cls = interpolate.byInterpolate(results, detections, tracker, frame, vehicle_clases, trajectories, angle_threshold, min_trajectory_length,counted_objects,counts,epsilon, giro_threshold)

                    
        
                
        #  mostrar el frame con los resultados
        cv2.imshow("Aforo direccional", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Conteos:")
    for cls, dir_counts in counts.items():
        print(f"{cls}:")
        for dir, count in dir_counts.items():
            print(f"{dir}: {count}")

    # mostrar resultados finales
