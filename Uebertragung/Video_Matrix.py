import time

from PIL import Image, ImageDraw, ImageFont
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

import cv2
import numpy as np


serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)
device.contrast(10)


video_path = "output_32x32.mp4"   # Pfad zu deinem Video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Konnte Video nicht öffnen")

while True:
    ret, frame = cap.read()   # ret = False, wenn keine Frames mehr da sind
    if not ret:
        break

    # mit 'q' abbrechen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    zweiteReihe = frame[8:16, :]
    zweiteReiheRot = np.rot90(zweiteReihe, 2)
    vierteReihe = frame[24:32, :]
    vierteReiheRot = np.rot90(vierteReihe, 2)

    frame[8:16, :] = zweiteReiheRot
    frame[24:32, :] = vierteReiheRot
    cv2.imshow('frame', frame)
    device.display(Image.fromarray(frame))
cap.release()
cv2.destroyAllWindows()
