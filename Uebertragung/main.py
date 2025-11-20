import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=-90)
device.contrast(10)

lst = "Das Crazy euda wir fahrn zu WM"
index = 0

while True:
    with canvas(device) as draw:
        # Zentrierung: x=2,y=1 für jedes Modul (optimale Werte für LCD_FONT)
        text(draw, (2, 1), lst[(index+0)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (10, 1), lst[(index+1)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (18, 1), lst[(index+2)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (26, 1), lst[(index+3)%len(lst)], fill="white", font=proportional(LCD_FONT))
    time.sleep(0.1)
    index += 1

