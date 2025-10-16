import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (1280, 720)})
cam.configure(config)
cam.start()

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=50, detectShadows=False)

# For smoothing motion detection
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    frame = cam.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Remove noise and small blobs
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Find motion contours (likely people)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for "detected people"
    motion_mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # ignore tiny noise, adjust threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(motion_mask, (x, y), (x + w, y + h), 255, -1)

    # Run Canny edge detection on entire frame
    edges = cv2.Canny(gray, 25, 75)

    # Keep only edges within the motion mask
    person_edges = cv2.bitwise_and(edges, edges, mask=motion_mask)

    # Optional: Overlay
    edge_bgr = cv2.cvtColor(person_edges, cv2.COLOR_GRAY2BGR)
    outlined = cv2.addWeighted(frame, 0.7, edge_bgr, 1.0, 0)

    cv2.imshow("Person Edges (filtered)", outlined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
