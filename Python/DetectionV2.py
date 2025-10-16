import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

while True:
    frame = cam.capture_array()

    # Ensure it’s 3-channel BGR
    if frame.ndim == 2:  # grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # sometimes RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (6, 6), 1.4)

    edges = cv2.Canny(blurred, 60, 150)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Resize to match just in case (avoid size mismatch error)
    edge_bgr = cv2.resize(edge_bgr, (frame.shape[1], frame.shape[0]))

    outlined = cv2.addWeighted(frame, 0.8, edge_bgr, 0.7, 0)

    cv2.imshow("Edges", edge_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
