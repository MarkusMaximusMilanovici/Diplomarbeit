import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

# Initialisierung: 4 Module hintereinander ("cascaded=4" bedeutet 4x8=32 LEDs Breite)
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=0)

while True:
    with canvas(device) as draw:
        # Beispiel-Muster: alle 4 Module durch eine Linie trennen
        draw.line((0, 0, 31, 0), fill="white")    # obere Linie über alle 4 Module
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        # Einzelne Pixel als Demo setzen
        draw.point((3, 5), fill="white")
        draw.point((10, 2), fill="white")
        draw.point((20, 5), fill="white")
        draw.point((28, 7), fill="white")
    time.sleep(0.1)  # Update-Intervall, kann angepasst werden
