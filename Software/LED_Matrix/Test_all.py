from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas
import time

# 👉 HIER nur Anzahl deiner Module ändern
NUM_MODULES = 32   # z.B. 18 Module

# SPI
serial = spi(port=0, device=0, gpio=noop())

# Device (nur Anzahl zählt!)
device = max7219(serial, cascaded=NUM_MODULES)

# Helligkeit
device.contrast(10)

# ALLE LEDs AN
with canvas(device) as draw:
    draw.rectangle(device.bounding_box, fill="white")

# Optional: dauerhaft an lassen
while True:
    time.sleep(1)