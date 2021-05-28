import os
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import dlib
from architecture import *
import cv2
import random
import time


face_data = 'train_img'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
encoder_weights_path = 'facenet_keras_weights.h5'
print('Initializing Face Detector Object...')
face_detector = dlib.get_frontal_face_detector()
print('Face Detector Object Initialized.')
print('Loading Weights into FaceNet...')
face_encoder.load_weights(encoder_weights_path)
print('Weights Loaded into FaceNet.')
encoding_dict = dict()
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def train_FaceNet_using_HOG():
    start = time.time()
    for face_names in os.listdir(face_data):
        person_dir = os.path.join(face_data, face_names)
        encodes = []
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img_BGR = cv2.imread(image_path)
            img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
            rects = face_detector(img_GRAY, 0)
            print(len(rects))
            if len(rects) == 0:
                continue
            rect = rects[0]
            x1, y1, width, height = rect.left(), rect.top(), rect.width(), rect.height()
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)[y1: y2, x1: x2]
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)
        if encodes:
            encode = np.sum(encodes, axis = 0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis = 0))[0]
            encoding_dict[face_names] = encode
            print(str(face_names), ' appended!')
    end = time.time()
    seconds = end - start
    print(seconds)
    path = 'encoding/encodings_HOG.pkl'
    with open(path, 'wb') as file:
        pickle.dump(encoding_dict, file)

if __name__ == '__main__':
    train_FaceNet_using_HOG()


