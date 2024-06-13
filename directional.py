import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import deepSort_method as dpM
import interpolate_method as interpolate
from scipy.interpolate import CubicSpline
from fastkml import kml
from shapely.geometry import LineString, Point
import math
import xml.etree.ElementTree as ET
import pyproj
 # funcion para realizar el metodo por deepSort (no tan eficiente sin modelo cnn)
 # cls = dpM.deepSortMethod(results, detections, tracker, frame,CONFIDENCE_THRESHOLD, vehicle_clases, trajectories, angle_threshold,min_trajectory_length,counted_objects,counts)
 # # funcion metodo deepSORT y Interpolacion
 #cls = interpolate.byInterpolate(results, detections, tracker, frame, vehicle_clases, trajectories, angle_threshold, min_trajectory_length,counted_objects,counts,epsilon, giro_threshold)


CONFIDENCE_THRESHOLD = 0.5  # nivel de confianza
try:
    import torch

    GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU")
except ModuleNotFoundError:
    GPU = False


def id_detection_directional():
    # modelo yoloV8 la version mas estable y precisa

    model = YOLO("yolov8n.pt") # carga del modelo

    model.fuse()  # mejoras las capas del modelo para un mejor rendimiento

    min_trajectory_length = 5  # establece el número mínimo de puntos en la trayectoria de un objeto que se deben considerar para calcular la curvatura y determinar si está girando
    # definir vehiculos y contadores
    vehicle_clases = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","unknown",]
    directions = ["Norte", "Sur", "Este", "Oeste", "detenido", 'giro_derecha', 'giro_izquierda', 'vuelta_u','Sureste', 'Suroeste', 'Noreste', 'Noroeste']  # diccionario de direcciones a tomar en cuenta
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
    
    # leer archivo kml y extraer coordenadas de los puntos de referencia
    tree = ET.parse('route2.kml')
    root = tree.getroot()
    placemarks = root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")
    
    reference_points = []
    for placemark in placemarks:
        point = placemark.find('.//{http://www.opengis.net/kml/2.2}Point')
        if point is not None:
            coordinates_str = point.find('.//{http://www.opengis.net/kml/2.2}coordinates').text
            lon,lat, _ = map(float, coordinates_str.split(','))
            reference_points.append((lon,lat))
    
    # Verificar que haya dos puntos de referencia
    if len(reference_points) != 2:
        raise ValueError("El archivo KML debe contener exactamente dos puntos de referencia.")        
    
    #convertir puntos de referencia a coordenadas de la imagen
    reference_points_image = [tuple(map(int, point)) for point in reference_points]
    
    # Diccionario para almacenar las transiciones de cuadrantes de cada objeto
    quadrant_transitions = {}
    
    # Diccionario para almacenar la dirección de cada objeto
    track_directions = {}  # Inicializar el diccionario aquí
    
    # Calcular la orientación (ángulo) del eje norte-sur
    orientation = math.degrees(math.atan2(
        reference_points_image[1][1] - reference_points_image[0][1],
        reference_points_image[1][0] - reference_points_image[0][0]
    ))
    print(orientation)
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
      
        if frame is None:
            continue
        detections = []
        # deteccion de vehiculos
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > CONFIDENCE_THRESHOLD: # nivel de confianza
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro delimitador
                cls = int(box.cls[0])  # Obtener la clase del vehiculo
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

        # Actualizar seguimiento con Deep SORT
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update >1:
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1,y1,x2,y2 = map(int, bbox)
            
            if 0<= track.det_class < len(vehicle_clases):
                cls = vehicle_clases[track.det_class]
            else:
                cls = 'unknown'
            
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))
            
            # Inicializar puntos con valores predeterminados
            prev_point = np.array([center_x, center_y])
            curr_point = np.array([center_x, center_y])

            # Interpolación cúbica de la trayectoria
            interpolated_points = []
            if len(trajectories[track_id]) >= 4:
                x = [p[0] for p in trajectories[track_id]]
                y = [p[1] for p in trajectories[track_id]]
                cs_x = CubicSpline(range(len(x)), x)
                cs_y = CubicSpline(range(len(y)), y)

                # Generar puntos interpolados (asegurando que sean enteros)
                num_points = 20
                for i in range(len(x) - 1):
                    t = np.linspace(i, i + 1, num_points)
                    interpolated_x = cs_x(t)
                    interpolated_y = cs_y(t)
                    interpolated_points.extend(zip(map(int, interpolated_x), map(int, interpolated_y)))
            
            # Asignar valores a prev_point y curr_point
            if len(interpolated_points) >= 2:
                prev_point = np.array(interpolated_points[-2])
                curr_point = np.array(interpolated_points[-1])
            elif len(trajectories[track_id]) >= 2:
                prev_point = np.array(trajectories[track_id][-2])
                curr_point = np.array(trajectories[track_id][-1])
            
            # Asignar cuadrante al objeto usando el punto de referencia
            quadrant = get_quadrant(center_x, center_y, reference_points_image,orientation)
            
            if track_id not in quadrant_transitions:
                quadrant_transitions[track_id] = []
            if quadrant_transitions[track_id] and quadrant_transitions[track_id][-1] != quadrant:
                # Se ha producido una transición de cuadrante
                prev_quadrant = quadrant_transitions[track_id][-1]
                if prev_quadrant == 'Norte' and quadrant == 'Sur':
                    # El objeto ha pasado de norte a sur
                    print(f"Objeto {track_id} ({cls}) ha pasado de Norte a Sur ")
                    # Guardar el evento en un archivo o base de datos (opcional)
                # ... (puedes agregar más condiciones para otras transiciones de interés)
            quadrant_transitions[track_id].append(quadrant)  # Actualizar el historial de cuadrantes
            
            direction = get_quadrant(center_x,center_y, reference_points_image,orientation )
            
            if direction not in ['Norte', 'Sur', 'Este', 'Oeste']:
                if len(trajectories[track_id]) >= 3:  # Necesitamos al menos 3 puntos para calcular el ángulo de giro
                    p1 = np.array(trajectories[track_id][-3])
                    p2 = np.array(trajectories[track_id][-2])
                    p3 = np.array(trajectories[track_id][-1])

                    v1 = p2 - p1
                    v2 = p3 - p2

                    # Calcular el ángulo de giro
                    angle = np.degrees(np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))

                    if abs(angle) > giro_threshold:
                        if angle > 0:
                            direction = 'giro_derecha'
                    else:
                            direction = 'giro_izquierda'

                    # Detectar vuelta en U (opcional)
                    if len(trajectories[track_id]) >= 5 and abs(angle) > 150:
                        direction = 'vuelta_u'

            track_directions[track_id] = direction
                    
            # conteo de los objetos
            if track_id not in counted_objects:
                counts[cls][direction] += 1
                counted_objects[track_id] = True
        
             
            # Dibujar cuadro delimitador, etiqueta y trayectoria interpolada
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar en el frame original
            cv2.putText(frame, f'{cls} - {direction} - ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Dibujar trayectoria interpolada
            if len(interpolated_points) > 1:
                for i in range(1, len(interpolated_points)):
                    # Convertir a enteros justo antes de dibujar la línea
                    start_point = tuple(map(int, interpolated_points[i - 1]))
                    end_point = tuple(map(int, interpolated_points[i]))

                    # Verificar límites de la imagen para los puntos interpolados
                    if (0 <= start_point[0] < frame.shape[1] and
                        0 <= start_point[1] < frame.shape[0] and
                        0 <= end_point[0] < frame.shape[1] and
                        0 <= end_point[1] < frame.shape[0]):

                        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            cv2.line(frame, reference_points_image[0], reference_points_image[1], (0, 255, 255), 2)  # Línea amarilla
            cv2.putText(frame, f'Orientacion: {orientation:.2f} grados', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        #  mostrar el frame con los resultados
        cv2.imshow("Aforo direccional", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Conteos:")
    for cls, dir_counts in counts.items():
        for dir, count in dir_counts.items():
            print(f"{dir}: {count}")

    # mostrar resultados finales


def get_quadrant(x, y, reference_point_image, orientation):
    ref_x, ref_y = reference_point_image

    # Ajustar la orientación para que el Norte esté en la parte superior
    orientation = (orientation - 90) % 360
    
    print(ref_x)
    print(ref_y)
    # Crear una línea que represente el eje Norte-Sur
    north_point = [ref_x[0], ref_y[1] - 100]  # Punto 100 píxeles al norte del punto de referencia
    south_point = [ref_x[0], ref_y[1] + 100]  # Punto 100 píxeles al sur del punto de referencia
    # Crear un tercer punto desplazado ligeramente para evitar el problema de Shapely
    #slightly_shifted_south_point = (south_point[0] + 0.0001, south_point[1])  
    

    ref_line = LineString([[north_point[0], south_point[0]],[north_point[1], south_point[0]],[north_point[0], south_point[1]]])

    # Crear un punto con las coordenadas del objeto
    point = Point(x, y)

    # Verificar si el punto está cerca de la línea de referencia
    if ref_line.distance(point) < 5:  # Manejar casos donde el objeto está muy cerca de la línea de referencia
        return 'detenido'

    # Determinar el cuadrante en función de la posición del punto y la orientación
    angle_with_ref = math.degrees(math.atan2(y - ref_y[1], x - ref_x[1]))
    angle_with_ref = (angle_with_ref - orientation) % 360  # Ajustar el ángulo al sistema de coordenadas de la cámara

    if 0 <= angle_with_ref < 90:
        return 'Noreste'
    elif 90 <= angle_with_ref < 180:
        return 'Sureste'
    elif 180 <= angle_with_ref < 270:
        return 'Suroeste'
    else:
        return 'Noroeste'