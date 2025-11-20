import time
from PIL import Image, ImageDraw, ImageFont
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)
device.contrast(10)

lst = "Das        Crazy        euda        wir        fahrn        zur        WM    "
index = 0

pil_font = ImageFont.load_default()

def draw_rotated_char(draw, x, y, char, font, fill):
    im = Image.new("1", (8, 8))
    idraw = ImageDraw.Draw(im)
    idraw.text((2, -1), char, font=font, fill=fill)  # Y hier auf -1, ggf. weiter anpassen
    im = im.rotate(180)
    draw.bitmap((x, y), im, fill=fill)

while True:
    with canvas(device) as draw:
        for block in range(16):
            grp = block // 4
            pos_in_grp = block % 4
            char = lst[(index + block) % len(lst)]
            if grp % 2 == 0:
                x = 2 + 8 * block
                text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
            else:
                block_start = 8 * (grp * 4)
                x = block_start + (8 * (3 - pos_in_grp)) + 2
                draw_rotated_char(draw, x, 0, char, pil_font, "white")
    #time.sleep(0.01)
    index += 1
