import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# --- Background subtractor (detects motion / people) ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=10, detectShadows=False)

while True:
    frame = cam.capture_array()

    # Ensure BGR format
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction → mask of moving objects (likely person)
    fgmask = fgbg.apply(gray)

    # Display the mask directly (white = detected person/motion)
    cv2.imshow("Person Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
