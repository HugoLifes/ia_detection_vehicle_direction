import cv2
import os
import numpy as np
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.8  # nivel de confianza
try:
    import torch

    GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU")
except ModuleNotFoundError:
    GPU = False


def id_detection_directional():
    # modelo yoloV8 la version mas estable y precisa

    model = YOLO('yolov8n.pt')
    model.fuse()
    # cnn_model = tf.keras.models.load_model('yolov8n.h5')

    # definir vehiculos y contadores
    vehicle_clases = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
    entrada_counts = {cls: 0 for cls in vehicle_clases}
    salida_counts = {cls: 0 for cls in vehicle_clases}

    cap = cv2.VideoCapture('video.mp4')  # captura video

    angle_treshold = 90  # umbral de cambio de angulo
    distance_treshold = 100  # umbral de distancia minima recorrida por pixel
    max_age = 40  # numero maximo de frames que un objeto puede desaparecer antes de ser eliminado
    min_hits = 3  # numero minimo de detecciones antes de que un objeto se considere confirmado
    trajectories = {}  # diccionario para almacenar la trayectoria de los objetos
    tracker = DeepSort(max_age=max_age, embedder_gpu=GPU)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = []
        # deteccion de vehiculos
        results = model(frame)[0]

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas del cuadro delimitador

                cls = vehicle_clases[int(box.cls[0])]  # Obtener la clase del vehiculo

                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
        # [x1, y1, x2 - x1, y2 - y1]
        tracks = tracker.update_tracks(detections, frame=frame)  # actualiza el movimiento
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            cls = track.det_class

            # almacenar trayectoria
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2))  # Centro del cuadro delimitador

            # predecir vuelta
            if len(trajectories[track_id]) > 10:  # logica para predecir vuelta
                prev_point = np.array(trajectories[track_id][-2])
                curr_point = np.array(trajectories[track_id][-1])

                # verificar si el vehiculo se ha movido lo suficiente
                distance = np.linalg.norm(prev_point - curr_point)
                if distance < distance_treshold:
                    continue

                if len(trajectories[track_id]) >= 3:
                    prev_prev_point = np.array(trajectories[track_id][-3])
                    prev_direction = prev_prev_point - prev_prev_point
                    curr_direction = curr_point - prev_point
                    angle_change = np.degrees(np.arccos(np.dot(prev_direction, curr_direction) / (
                            np.linalg.norm(prev_direction) * np.linalg.norm(curr_direction))))

                    if angle_change > angle_treshold:
                        print(f'Vehiculo {track_id} dio una vuelta')

            # clasificar la direccion con el modelo cnn
            if track.time_since_update > 10:
                features = np.array([x1, y1, x2, y2], track.conf)
                direction = model.predict(features.reshape(1, -1))[0]
                direction = 'entrada' if direction[0] > direction[1] else 'salida'

                if direction == 'entrada':
                    entrada_counts[cls] += 1
                else:
                    salida_counts[cls] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls} - ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        #  mostrar el frame con los resultados
        cv2.imshow('Aforo direccional', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # mostrar resultados finales
    print('Conteos de entrada')
    for cls, count in entrada_counts.items():
        print(f'{cls}: {count}')

    print('Conteos de salida')
    for cls, count in salida_counts.items():
        print(f'{cls}: {count}')