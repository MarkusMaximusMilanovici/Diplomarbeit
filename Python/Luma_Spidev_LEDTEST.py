from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas
import time

serial = spi(port=0, device=0, gpio=noop())  # <-- use the classic one
device = max7219(serial, cascaded=3)
device.contrast(255)

with canvas(device) as draw:
    draw.rectangle(device.bounding_box, fill="white")

time.sleep(3)
device.clear()
