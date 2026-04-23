from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas

# SPI initialisieren
serial = spi(port=0, device=0, gpio=noop())

# Device (deine Matrix)
device = max7219(serial, block_orientation=-90, width=32, height=48)

# Helligkeit (0–255)
device.contrast(10)

# Alle LEDs einschalten
with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white", fill="white")