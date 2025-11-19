import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

# 4 Module hintereinander
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=-90)

device.contrast(120)

while True:
    with canvas(device) as draw:
        # Jeder Buchstabe auf die entsprechende Matrix (8 Pixel Schritte)
        text(draw, (0, 0), "M", fill="white", font=proportional(LCD_FONT))
        text(draw, (8, 0), "A", fill="white", font=proportional(LCD_FONT))
        text(draw, (16, 0), "N", fill="white", font=proportional(LCD_FONT))
        text(draw, (24, 0), "U", fill="white", font=proportional(LCD_FONT))
    time.sleep(0.2)
