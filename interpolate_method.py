import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from fastkml import kml
from shapely.geometry import LineString
import math
def byInterpolate(results, detections, tracker, frame, vehicle_clases, trajectories, angle_threshold, min_trajectory_length,counted_objects,counts,epsilon, giro_threshold):
        # Leer archivo KML
        with open("prueba.kml", 'rb') as f:
            content = f.read()
            content = content.decode('utf-8')
            k = kml.KML()
            k.from_string(content)

        # Extraer coordenadas de la trayectoria KML
        features = list(k.features())
        placemark = features[0]
        line_string = placemark.geometry  # Asumimos que el KML contiene una LineString

        # Calcular el centro de la trayectoria KML
        coords = list(line_string.coords)
        center_x, center_y = np.mean(coords, axis=0)

        # Calcular la orientación (ángulo) de la trayectoria KML
        line = LineString(coords)
        orientation = math.degrees(math.atan2(line.coords[1][1] - line.coords[0][1], line.coords[1][0] - line.coords[0][0]))

        # Diccionario para almacenar las transiciones de cuadrantes de cada objeto
        quadrant_transitions = {}
        
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro delimitador
                cls = int(box.cls[0])  # Obtener la clase del vehiculo
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

        # Actualizar seguimiento con Deep SORT
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Verificar si el índice está dentro de los límites de la lista
            if 0 <= track.det_class < len(vehicle_clases):
                cls = vehicle_clases[track.det_class]
            else:
                cls = "unknown"

            # Almacenar trayectoria (usando las coordenadas originales)
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
                num_points = 10
                # Generar puntos interpolados
                for i in range(len(x) - 1):
                    t = np.linspace(i, i + 1, num_points)
                    interpolated_x = cs_x(t)
                    interpolated_y = cs_y(t)
                    interpolated_points.extend(zip(map(int, interpolated_x), map(int, interpolated_y)))

            
            if len(interpolated_points) >= 2:  # Usar puntos interpolados si existen
                prev_point = np.array(interpolated_points[-2])
                curr_point = np.array(interpolated_points[-1])
                
            elif len(trajectories[track_id]) >= 2 :  # Usar puntos originales si no hay suficientes para interpolar
                prev_point = np.array(trajectories[track_id][-2])
                curr_point = np.array(trajectories[track_id][-1])
            else:
                continue
            
            # Determinar dirección y giro (usando get_quadrant())
            direction = get_quadrant(center_x, center_y, orientation, frame, track)
            
            # Determinar dirección y giro (usando los puntos interpolados)
            if direction not in ['Norte', 'Sur', 'Este', 'Oeste']:
                delta_x = curr_point[0] - prev_point[0]
                delta_y = curr_point[1] - prev_point[1]
            
                angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
            
                if abs(angle) > angle_threshold:
                    direction = 'abajo' if delta_y < 0 else 'arriba'
                else: 
                    if len(trajectories[track_id]) >= min_trajectory_length:
                        points = np.array(trajectories[track_id][-min_trajectory_length:])
                        x = points[:,0]
                        y = points[:,1]
                    
                        # calcular la curvatura (usando diferencias finitas)
                        dx_dt = np.gradient(x)
                        dy_dt = np.gradient(y)
                        d2x_dt2 = np.gradient(dx_dt)
                        d2y_dt2 = np.gradient(dy_dt)
                        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt**2 + dy_dt**2)**1.5 + epsilon)
                    
                    #promediar la curvatura
                        avg_curvature = np.mean(curvature[-3:])
                    
                        if avg_curvature >= giro_threshold:
                            direction = 'giro'
                    else:
                        if abs(delta_x) > abs(delta_y):
                            direction = 'izquierda' if delta_x > 0 else 'derecha'
                
                ### detectar transiciones por cuadrante
                if track_id not in quadrant_transitions:
                    quadrant_transitions[track_id]= []
                if quadrant_transitions[track_id] and quadrant_transitions[track_id][-1] != direction:
                    prev_quadrant = quadrant_transitions[track_id][-1]
                    if prev_quadrant == 'Norte' and direction == 'Sur':
                        print(f"Objeto {track_id} ({cls}) ha pasado de Norte a Sur")
                    ## agregar mas direcciones de interes

                quadrant_transitions[track_id].append(direction)
                
                
            # Contar el objeto solo si no ha sido contado previamente
            if track_id not in counted_objects:
                counts[cls][direction] += 1
                counted_objects[track_id] = True
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls} - {direction} - ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Limpiar el diccionario de objetos contados al final del fotograma
        for track_id in list(counted_objects.keys()):
            if track_id not in [t.track_id for t in tracks]:
                del counted_objects[track_id]
                
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
        return cls



# Definir los cuadrantes en función del centro y la orientación del KML
def get_quadrant(center_x, center_y, orientation, frame, track):
    x, y = track.kf.x.flatten()[:2]  # Obtener coordenadas predichas por Kalman
    if y < center_y:
        if x < center_x:
            return 'Norte' if orientation < 45 or orientation >= 315 else 'Oeste'
        else:
            return 'Este' if 45 <= orientation < 135 else 'Norte'
    else:
        if x < center_x:
            return 'Oeste' if 225 <= orientation < 315 else 'Sur'
        else:
            return 'Sur' if 135 <= orientation < 225 else 'Este'