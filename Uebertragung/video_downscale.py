import cv2
from PIL import Image
import numpy as np

input_path = 'bad-apple.mp4'
output_path = 'output_32x32.mp4'

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (32, 32), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop to square (center crop)
    h, w, _ = frame.shape
    min_side = min(h, w)
    start_x = (w - min_side) // 2
    start_y = (h - min_side) // 2
    cropped = frame[start_y:start_y + min_side, start_x:start_x + min_side]

    # Resize to 32x32
    resized = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)

    # Convert to PIL and make black & white
    img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    bw = img.convert('1')

    # Convert PIL mode '1' back to numpy for OpenCV
    bw_array = np.array(bw, dtype=np.uint8)
    bw_array *= 255  # Mode '1' hat Werte 0 und 1, konvertiere zu 0 und 255

    out.write(bw_array)

cap.release()
out.release()
