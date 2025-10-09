from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_still_configuration()

native_width, native_height = config["main"]["size"]
print(native_width, native_height)

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Module 3 Wide NoIR", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
