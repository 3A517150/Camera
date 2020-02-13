from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

#設定攝影機
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

#新視窗命名為Faces
display_window = cv2.namedWindow("Faces")

#載入臉部特徵模組
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array

    #臉部辨識
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    #輸出影像
    cv2.imshow("Faces", image)

    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == ord("q"): # Press 'q' to quit
        break
