import cv2
import numpy as np
from picamera2 import Picamera2

# --- Initialize camera ---
cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# --- Background subtractor (detects motion / people) ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=10, detectShadows=False)

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

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Edge detection (on the whole frame)
    edges = cv2.Canny(gray, 25, 75)

    # Keep only edges inside the moving region
    person_edges = cv2.bitwise_and(edges, edges, mask=fgmask)

    # Convert edges to color for display
    edge_bgr = cv2.cvtColor(person_edges, cv2.COLOR_GRAY2BGR)

    # Optional: overlay edges on the original frame
    outlined = cv2.addWeighted(frame, 0.8, edge_bgr, 0.8, 0)

    cv2.imshow("Edges Only", person_edges)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
