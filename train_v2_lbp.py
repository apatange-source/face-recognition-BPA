from architecture import *
import os
import cv2
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import time

face_data = 'train_img'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = 'facenet_keras_weights.h5'
face_encoder.load_weights(path)
cascPath = 'lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

start = time.time()
for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data, face_names)
    encodes = []
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        img_BGR = cv2.imread(image_path)
        img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img_GRAY,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(100, 100))
        print(len(faces))
        if (len(faces) == 0):
            continue
        (x1, y1, width, height) = faces[0]
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)[y1: y2, x1: x2]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)

        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    print(str(face_names), ' appended')
    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode

end = time.time()
seconds = end - start
print(seconds)
path = 'encoding/encodings_LBP.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)



