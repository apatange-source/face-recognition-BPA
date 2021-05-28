import dlib
import cv2
import argparse
import os
import time
import mtcnn


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

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def face_detection_realtime():
    face_detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(r'C:\Users\aishw\PycharmProjects\FaceRecognition\WhatsApp Video 2021-05-19 at 9.03.38 PM.mp4')
    confidence_t = 0.99
    while True:
        start = time.time()
        _, image = cap.read()
        results = face_detector.detect_faces(image)
        for res in results:
            if res['confidence'] < confidence_t:
                continue
            face, pt_1, pt_2 = get_face(image, res['box'])
            draw_fancy_box(image, pt_1, pt_2, (127, 255, 255), 2, 10, 20)
            cv2.putText(image, "Face", (pt_1[0] - 20, pt_1[1] - 20),
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
    print("Real time face detection is starting ... ")
    face_detection_realtime()