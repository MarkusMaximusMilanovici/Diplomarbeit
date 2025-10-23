#sudo apt install python3-pip
#pip3 install luma.led_matrix

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from PIL import ImageDraw

serial = spi(port=10, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=90, rotate=0)

with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white", fill="white")
    draw.rectangle(device.bounding_box, outline="white", fill="white")
