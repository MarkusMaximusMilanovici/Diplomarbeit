import cv2
from PIL import Image
import numpy as np

# Input and output paths
input_path = 'bad-apple.mp4'
output_path = 'output_32x32.mp4'

# Open the video
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (32, 32))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop to square (center crop)
    h, w, _ = frame.shape
    min_side = min(h, w)
    start_x = (w - min_side) // 2
    start_y = (h - min_side) // 2
    cropped = frame[start_y:start_y+min_side, start_x:start_x+min_side]

    # Resize to 32x32
    resized = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)

    out.write(resized)

cap.release()
out.release()
