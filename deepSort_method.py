import numpy as np
import cv2
def deepSortMethod(results, detections, tracker, frame,CONFIDENCE_THRESHOLD, vehicle_clases, trajectories, angle_threshold,min_trajectory_length,counted_objects,counts):
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > CONFIDENCE_THRESHOLD: # nivel de confianza
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro delimitador
                cls = int(box.cls[0])  # Obtener la clase del vehiculo
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
                    
        # [x1, y1, x2 - x1, y2 - y1]
        tracks = tracker.update_tracks(detections, frame=frame)  # actualiza el movimiento
    
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            
            if 0 <= track.det_class < len(vehicle_clases):
                cls = vehicle_clases[track.det_class]
            else:
                cls = 'unknown'

            # almacenar trayectoria
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))  # Centro del cuadro delimitador

            direction = 'detenido' #direccion por defecto
            if len(trajectories[track_id])>=2:
                #features = np.array([x1, y1, x2, y2, track.get_det_conf()])
                prev_point = np.array(trajectories[track_id][-2])
                curr_point = np.array(trajectories[track_id][-1])
                delta_x = curr_point[0] - prev_point[0]
                delta_y = curr_point[1] - prev_point[1]

                angle = np.arctan2(delta_y, delta_x) * 180 / np.pi 

                if abs(angle) > angle_threshold:
                    direction = 'arriba' if delta_y < 0 else 'abajo'
                else:
                    if abs(delta_x) > abs(delta_y):
                        direction = 'derecha' if delta_x > 0 else 'izquierda'
                    else:
                        direction = 'detenido'
            # predecir vuelta
            if len(trajectories[track_id]) > 2:  # logica para predecir vuelta
                points = np.array(trajectories[track_id][-min_trajectory_length:])
                x = points[:, 0]
                y = points[:, 1]

                # Calcular la curvatura (usando diferencias finitas)
                dx_dt = np.gradient(x)
                dy_dt = np.gradient(y)
                d2x_dt2 = np.gradient(dx_dt)
                d2y_dt2 = np.gradient(dy_dt)
                curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5

                # Promediar la curvatura en los últimos puntos
                avg_curvature = np.mean(curvature[-3:])  # Promedio de los últimos 3 puntos

                if avg_curvature > angle_threshold:
                    direction = 'giro'
            # Actualizar contadores
            if track_id not in counted_objects:
                counts[cls][direction] += 1
                counted_objects[track_id] = True
        # Limpiar el diccionario de objetos contados al final del fotograma
        for track_id in list(counted_objects.keys()):
            if track_id not in [t.track_id for t in tracks]:
                del counted_objects[track_id]   
                   
            if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and 0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]:    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls} - {direction} -ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
        
        
        return cls