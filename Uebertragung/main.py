import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())

# Erstes 4er-Cluster, block_orientation -90
device1 = max7219(serial, cascaded=4, block_orientation=-90)
device1.contrast(10)

# Zweites 4er-Cluster, block_orientation 90
device2 = max7219(serial, cascaded=4, block_orientation=90)
device2.contrast(10)

lst = "Das Crazy euda wir fahrn zu WM "
index = 0

while True:
    # Erstes Modul schreiben: Buchstaben 0-3 auf device1
    with canvas(device1) as draw:
        text(draw, (2, 1), lst[(index + 0) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (10, 1), lst[(index + 1) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (18, 1), lst[(index + 2) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (26, 1), lst[(index + 3) % len(lst)], fill="white", font=proportional(LCD_FONT))
    # Zweites Modul schreiben: Buchstaben 4-7 auf device2
    with canvas(device2) as draw:
        text(draw, (2, 1), lst[(index + 4) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (10, 1), lst[(index + 5) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (18, 1), lst[(index + 6) % len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (26, 1), lst[(index + 7) % len(lst)], fill="white", font=proportional(LCD_FONT))
    time.sleep(0.5)
    index += 1
