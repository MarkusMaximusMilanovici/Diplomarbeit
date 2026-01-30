# ImagetoMatrix.py
import sys
import cv2
import numpy as np
from PIL import Image

RUN_ON_PI = sys.platform.startswith("linux")

if RUN_ON_PI:
    from luma.led_matrix.device import max7219
    from luma.core.interface.serial import spi, noop

    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, block_orientation=-90, width=32, height=48)
    device.contrast(10)
else:
    device = None  # Platzhalter ohne Hardware


def drawImage(frame):
    # Auf dem Laptop: einfach nichts machen
    if device is None:
        return

    # Bei 40 Zeilen Höhe gibt es 5 Blöcke à 8 Zeilen (0-7, 8-15, 16-23, 24-31, 32-39)
    # 2. und 4. 8er-Zeilenblock drehen
    zweite = frame[8:16, :]
    vierte = frame[24:32, :]

    frame[8:16, :] = np.rot90(zweite, 2)
    frame[24:32, :] = np.rot90(vierte, 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray)
    img = img.convert(device.mode)

    device.display(img)
