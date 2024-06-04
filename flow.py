import cv2
import os
import tensorflow as tf
import numpy as np
from ultralytics import YOLO


SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0


def try_training():
    
    model = YOLO('yolov8n.pt')
    
    model.save('yolov8n.keras')
    
    yolo_load = tf.keras.models.load_model('yolov8n.keras')
    
    optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,)
    
    yolo_load.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")
    
    yolo_load.load_weights()
    
    cap = cv2.VideoCapture('video.mp4')
    
    yolo_load.predict(cap)
    
    
    
    