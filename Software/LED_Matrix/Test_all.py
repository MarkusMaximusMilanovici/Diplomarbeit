from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time

# Anzahl deiner Matrizen (z.B. 4 für ein Modul)
NUM_MATRICES = 4

# SPI Setup
serial = spi(port=0, device=0, gpio=noop())

# Device
device = max7219(serial, cascaded=NUM_MATRICES, block_orientation=-90)

# 🔥 SEHR NIEDRIGE HELLIGKEIT (0–255)
device.contrast(1)   # probier auch 0 oder 2

# Größe berechnen
WIDTH = NUM_MATRICES * 8
HEIGHT = 8

# Alle LEDs AN
with canvas(device) as draw:
    draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), fill="white")

# Programm läuft einfach weiter (damit es nicht beendet wird)
while True:
    time.sleep(1)