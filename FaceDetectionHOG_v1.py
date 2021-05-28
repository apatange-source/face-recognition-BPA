# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 04:52:59 2020

@author: Amul Patange
"""

import dlib
import cv2
import argparse
import os
import time


def write_to_disk(image, face_cordinates):
    for (x1, y1, w, h) in face_cordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)


def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2


    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def face_detection_realtime():

    cap = cv2.VideoCapture(0)

    while True:
        start = time.time()
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        face_cordinates = []
        for (i, rect) in enumerate(rects):
            x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
                1, rect.bottom() + 1, rect.width(), rect.height()
            draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

            face_cordinates.append((x1, y1, w, h))

            cv2.putText(image, "Face #{}".format(i + 1), (x1 - 20, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)
        end = time.time()
        seconds = end - start 
        fps = round(1.0 / seconds, 2)
        cv2.putText(image,  
                'FPS = ' + str(fps),  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4)
        cv2.imshow("Output", image)

        if cv2.waitKey(30) & 0xFF == ord('s'):
            write_to_disk(image, face_cordinates)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    print("Real time face detection is starting ... ")
    face_detection_realtime()
