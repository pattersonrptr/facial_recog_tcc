# import the necessary packages
from __future__ import print_function
from facial_recog import FaceRecognizer
from PIL import Image
from PIL import ImageTk

import tkinter as tki
import threading
import datetime
import imutils
import time
import cv2
import os


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event

        self.recognizer = FaceRecognizer()

        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # To bound and crop the detected face
        self.subface_x = 0  # X position
        self.subface_y = 0  # Y position
        self.subface_h = 0  # Height
        self.subface_w = 0  # Width

        # Facial detector variables
        self.detected_face_name = ''
        self.access_status = False
        self.distance = '0'
        self.take_snapshot = False

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        self.fontePadrao = ("Arial", 10)

        self.nomeLabel = tki.Label(self.root, text="Nome", font=self.fontePadrao)
        self.nomeLabel.pack()

        self.nome = tki.Entry(self.root)
        self.nome["width"] = 30
        self.nome["font"] = self.fontePadrao
        self.nome.pack()

        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        self.thread_recog = threading.Thread(target=self.facial_recognition)
        self.thread_recog.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):

        size = 4
        facedata = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

        # We load the xml file
        classifier = cv2.CascadeClassifier(facedata)

        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=600)

                self.frame = cv2.flip(self.frame, 1, 0)  # Flip to act as a mirror

                # Resize the image to speed up detection
                mini = cv2.resize(self.frame, (int(self.frame.shape[1] / size), int(self.frame.shape[0] / size)))

                faces = classifier.detectMultiScale(mini)

                for f in faces:

                    bounds = [v * size for v in f]

                    self.update_subface_bounds(bounds)

                    cv2.rectangle(
                        self.frame,
                        (self.subface_x, self.subface_y),
                        (self.subface_x + self.subface_w, self.subface_y + self.subface_h),
                        (0, 255, 0),
                        thickness=4
                    )

                    if self.access_status:
                        access_status = 'Permitido'
                        access_color = (0, 255, 0)
                    else:
                        access_status = 'Negado'
                        access_color = (0, 0, 255)

                    x, y, h, w = self.subface_x, self.subface_y, self.subface_h, self.subface_w

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.frame, self.detected_face_name, (x, y - 45), font, 0.8, (0, 140, 255))
                    cv2.putText(self.frame, access_status, (x, y - 15), font, .8, access_color)
                    cv2.putText(self.frame, self.distance, (x, y + h + 22), font, .8, (0, 140, 255))

                    bounds = (self.subface_x, self.subface_y, self.subface_h, self.subface_w)
                    self.capture_face_frame(bounds)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def facial_recognition(self):

        while True:
            time.sleep(.3)

            if self.recognizer.database:
                try:
                    access, min_dist, identity = self.recognizer.who_is_it("images/camera_0.jpg")
                    self.detected_face_name = identity.capitalize()
                    self.access_status = access
                    self.distance = f'dist: {min_dist}'

                except TypeError as e:
                    print('Imagem não encontrada:', str(e))

    def update_subface_bounds(self, bounds):

        (x, y, w, h) = bounds

        self.subface_w, self.subface_h = w + 70, h + 70
        n_x, n_y = x - 45, y - 45

        self.subface_x = n_x if n_x >= 0 else 0
        self.subface_y = n_y if n_y >= 0 else 0

    def capture_face_frame(self, bounds):
        # Save just the rectangle faces in SubRecFaces

        x, y, h, w = bounds
        sub_face = self.frame[y: y + h, x: x + w]

        dim = (96, 96)
        sub_face = cv2.resize(sub_face, dim, interpolation=cv2.INTER_LINEAR)

        face_file_img = './images/camera_0.jpg'

        cv2.imwrite(face_file_img, sub_face)

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        # ts = datetime.datetime.now()
        # filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        name = self.nome.get()

        filename = f'{name}.jpg'

        if not filename:
            print('Name field is empty.')
            return

        p = os.path.sep.join((self.outputPath, filename))

        frame_copy = self.frame.copy()

        # Save just the rectangle faces in sub_face
        sub_face = frame_copy[
                   self.subface_y: self.subface_y + self.subface_h,
                   self.subface_x: self.subface_x + self.subface_w
                   ]

        dim = (96, 96)

        sub_face = cv2.resize(sub_face, dim, interpolation=cv2.INTER_LINEAR)

        # save the file
        cv2.imwrite(p, sub_face)

        print('Inserindo', name, p)
        self.recognizer.insert_new_person(name, p)

        print(f"[INFO] saved {filename}")

        self.recognizer.database = self.recognizer.load_img_data()

        print('Pessoas na base', [k for k in self.recognizer.database])

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
