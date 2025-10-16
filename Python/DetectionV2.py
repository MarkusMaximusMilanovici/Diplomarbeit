import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

while True:
    frame = cam.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur smooths noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny edge detection
    edges = cv2.Canny(blurred, 60, 150)

    # Convert edges to 3-channel for overlay
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine original + edges
    outlined = cv2.addWeighted(frame, 0.8, edge_bgr, 0.7, 0)

    cv2.imshow("Edges", outlined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
