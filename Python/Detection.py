from picamera2 import Picamera2
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

vcap = cv2.VideoCapture()

picam2 = Picamera2()
config = picam2.create_still_configuration()

native_width, native_height = config["main"]["size"]
print(native_width, native_height)

picam2.start()

while True:

    frame = picam2.capture_array()

    frame[:, :, 2] = (frame[:, :, 2] * 0.5).astype(np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Camera Module 3 Wide NoIR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
