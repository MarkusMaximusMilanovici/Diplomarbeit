from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas
import time

# SPI initialisieren
serial = spi(port=0, device=0, gpio=noop())

# Device initialisieren
device = max7219(serial, block_orientation=-90, width=16, height=2)

# Helligkeit einstellen
device.contrast(10)

while True:
    # ALLE LEDs AN
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, fill="white")

    time.sleep(1)  # 1 Sekunde an

    # ALLE LEDs AUS
    device.clear()

    time.sleep(1)  # 1 Sekunde aus