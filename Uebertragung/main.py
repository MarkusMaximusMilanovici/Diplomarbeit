import time
from PIL import Image, ImageDraw
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

# SPI-Init und Matritzen-Setup
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)
device.contrast(10)

# Animierter Text
lst = "Das Crazy euda wir fahrn zu WM "
index = 0

def draw_rotated_char(draw, x, y, char, font, fill):
    # Ein 8x8 Bild nur für den Buchstaben zeichnen
    img = Image.new("1", (8, 8))
    idraw = ImageDraw.Draw(img)
    idraw.text((2, 1), char, font=font, fill=fill)
    img = img.rotate(180)
    draw.bitmap((x, y), img, fill=fill)

while True:
    with canvas(device) as draw:
        for block in range(16):
            grp = block // 4
            pos_in_grp = block % 4
            char = lst[(index + block) % len(lst)]
            x = 2 + 8 * block if grp % 2 == 0 else (8 * (grp * 4) + 8 * (3 - pos_in_grp) + 2)
            if grp % 2 == 0:
                text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
            else:
                draw_rotated_char(draw, x, 0, char, proportional(LCD_FONT), "white")
    time.sleep(0.5)
    index += 1
