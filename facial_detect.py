#! /usr/bin/python3

import cv2
import os

from threading import Thread
from shutil import copyfile


class FaceDetector(Thread):

    def __init__(self):

        Thread.__init__(self)

        self.detected_face_name = ''
        self.access_status = False
        self.distance = '0'
        self.take_snapshot = False

    def run(self):

        print('Doing facial recognition')

        size = 4
        webcam = cv2.VideoCapture(0)    # Use camera 0

        facedata = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

        # We load the xml file
        classifier = cv2.CascadeClassifier(facedata)

        while True:
            (rval, im) = webcam.read()
            im = cv2.flip(im, 1, 0)    # Flip to act as a mirror

            # Resize the image to speed up detection
            mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))

            # detect MultiScale / faces
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:

                (x, y, w, h) = [v * size for v in f]    # Scale the shapesize backup

                w, h = w + 70, h + 70

                n_x, n_y = x - 45, y - 45

                x = n_x if n_x >= 0 else 0
                y = n_y if n_y >= 0 else 0

                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)

                if self.access_status:
                    access_status = 'Permitido'
                    access_color = (0, 255, 0)
                else:
                    access_status = 'Negado'
                    access_color = (0, 0, 255)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im, self.detected_face_name, (x, y - 45), font, 0.8, (0, 140, 255))
                cv2.putText(im, access_status, (x, y - 15), font, .8, access_color)
                cv2.putText(im, self.distance, (x, y + h + 22), font, .8, (0, 140, 255))

                # Save just the rectangle faces in SubRecFaces
                sub_face = im[y: y + h, x: x + w]

                dim = (96, 96)

                sub_face = cv2.resize(sub_face, dim, interpolation=cv2.INTER_LINEAR)

                face_file_img = './images/camera_0.jpg'

                cv2.imwrite(face_file_img, sub_face)

            # Show the image
            cv2.imshow('Camera 0',   im)
            key = cv2.waitKey(10)

            # if Esc key is press then break out of the loop
            if key == 27:    # The Esc key
                break

            if self.take_snapshot:

                self.take_snapshot = False

                src = os.path.join(os.getcwd(), 'images/camera_0.jpg')
                dest = os.path.join(os.getcwd(), 'images/patterson.jpg')

                from facial_recog import insert_new_person

                copyfile(src, dest)
                insert_new_person('junior', dest)
