import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

# 4 Module hintereinander
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=-90)

while True:
    with canvas(device) as draw:
        # "MANU" auf die LED-Matrix schreiben, Startpunkt (0, 0)
        text(draw, (0, 0), "MANU", fill="white", font=proportional(LCD_FONT))
    time.sleep(0.2)  # Kurze Pause, Display bleibt trotzdem angezeigt
