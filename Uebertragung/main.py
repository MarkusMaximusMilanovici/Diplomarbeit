import time
from PIL import Image, ImageDraw
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
# Insgesamt 16 Blöcke (also 4x4er-Modul), block_orientation ggf. anpassen
device = max7219(serial, cascaded=16, block_orientation=-90)
device.contrast(10)

lst = "Das Crazy euda wir fahrn zu WM "
index = 0

def draw_rotated_char(draw, x, y, char, font, fill):
    im = Image.new("1", (8, 8))
    idraw = ImageDraw.Draw(im)
    idraw.text((2, 1), char, font=font, fill=fill)
    im = im.rotate(180)
    draw.bitmap((x, y), im, fill=fill)

while True:
    with canvas(device) as draw:
        for block in range(16):
            grp = block // 4            # Modulgruppe (0-3)
            pos_in_grp = block % 4      # Position im Modul
            char = lst[(index + block) % len(lst)]
            # X-Position im Serienverbund, immer 8 Pixel pro Block, 2 Pixel Offset
            if grp % 2 == 0:
                x = 2 + 8 * block
                text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
            else:
                # X ist gespiegelt für rückwärts-Modul
                block_start = 8 * (grp * 4)
                x = block_start + (8 * (3 - pos_in_grp)) + 2
                draw_rotated_char(draw, x, 0, char, proportional(LCD_FONT), "white")
    time.sleep(0.25)
    index += 1
