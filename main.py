# !/usr/bin/python3.6

import time
from application import Application

from facial_recog import FaceRecognizer
from facial_detect import FaceDetector
from tkinter import Tk


def facial_recognition():

    recognizer = FaceRecognizer()
    detector = FaceDetector()

    detector.start()

    while True:
        time.sleep(.3)

        try:
            access, min_dist, identity = recognizer.who_is_it("images/camera_0.jpg")
            detector.detected_face_name = identity.capitalize()
            detector.access_status = access
            detector.distance = f'dist: {min_dist}'

        except TypeError as e:
            print(str(e))


if __name__ == '__main__':

    root = Tk()
    Application(root)
    root.mainloop()

    # facial_recognition()
