import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_still_configuration()
native_width, native_height = config["main"]["size"]
cam.start()

mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

while True:
    frame = cam.capture_array()

    result = mp_seg.process(frame)
    mask = result.segmentation_mask
    human_mask = (mask > 0.5).astype(np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    human_edges = cv2.bitwise_and(edges, edges, mask=human_mask)

    outline = cv2.cvtColor(human_edges, cv2.COLOR_GRAY2BGR)
    outlined = cv2.addWeighted(frame, 0.8, outline, 0.7, 0)

    cv2.imshow("Type Shit", outlined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop()