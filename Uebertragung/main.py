import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)  # 16 Blöcke seriell

device.contrast(10)
lst = "Das Crazy euda wir fahrn zu WM "
index = 0

while True:
    with canvas(device) as draw:
        for block in range(16):
            block_group = block // 4      # 0,1,2,3 für jedes 4er-Modul
            pos_in_group = block % 4      # 0...3 innerhalb des Moduls

            if block_group % 2 == 0:
                # GERADE Zeile (z.B. ganz oben, dritte Modul-Reihe von oben): Links --> Rechts
                x = 2 + 8 * block
            else:
                # UNGERADE Zeile: Rechts --> Links (invertierte x-Position innerhalb des 4er-Moduls)
                block_start = 8 * (block_group * 4)
                x = block_start + (8 * (3 - pos_in_group)) + 2

            char = lst[(index + block) % len(lst)]
            text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
    time.sleep(0.5)
    index += 1
