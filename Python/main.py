from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Module 3 Wide NoIR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
