import time
from PIL import Image
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
import cv2
import numpy as np

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, block_orientation=90, width = 32, height = 32)
device.contrast(10)

print(device.width)
print(device.height)

video_path = "output_32x32.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Konnte Video nicht öffnen")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ESC / q zum Abbrechen (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 2. und 4. 8er-Zeilenblock drehen
    zweite = frame[8:16, :]
    vierte = frame[24:32, :]

    frame[8:16, :] = np.rot90(zweite, 2)
    frame[24:32, :] = np.rot90(vierte, 2)

    # Debug: im Fenster anzeigen
    cv2.imshow("frame", frame)

    # BGR -> GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for _ in range(4):

        # PIL-Image aus Graustufen machen
        img = Image.fromarray(gray)

        # Bildmodus an Device anpassen (oft "1" oder "L")
        img = img.convert(device.mode)

        device.display(img)
        time.sleep(0.5)
        gray = np.rot90(gray, 1)

cap.release()
cv2.destroyAllWindows()
