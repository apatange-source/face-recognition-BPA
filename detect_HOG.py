import os
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import dlib
from architecture import *
import cv2
from scipy.spatial.distance import cosine
import time
from train_v2_HOG import normalize, l2_normalizer

confidence_t = 0.99
recognition_t = 0.5
required_shape = (160, 160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, face_detector, face_encoder, encoding_dict):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(img_GRAY, 0)
    name = 'unknown'
    for rect in rects:
        name = 'unknown'
        box = (rect.left(), rect.top(), rect.width(), rect.height())
        face, pt_1, pt_2 = get_face(img_RGB, box)
        encode = get_encode(face_encoder, face, required_shape)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + ' ' + str(round(distance, 2)), (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            #cv2.putText(img, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return img, name

def identify_Using_FaceNet():
    encodings_path = 'encoding/encodings_HOG.pkl'
    encoding_dict = load_pickle(encodings_path)
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    face_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("CAM NOT OPENED")
            break
        frame, names = detect(frame, face_detector, face_encoder, encoding_dict)
        end = time.time()
        seconds = end - start
        fps = round(1.0 / seconds, 2)
        cv2.putText(frame, 'FPS = ' + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    identify_Using_FaceNet()