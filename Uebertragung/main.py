import time
from PIL import Image, ImageDraw
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)

device.contrast(10)
lst = "Das Crazy euda wir fahrn zu WM "
index = 0

def draw_rotated_text(draw, x, y, char, fill, font):
    # Erstelle ein leeres 8x8-Bild
    im = Image.new("1", (8, 8))
    idraw = ImageDraw.Draw(im)
    idraw.text((2, 1), char, font=font, fill=fill)
    # Drehe das Bild um 180°
    im = im.rotate(180)
    # Füge es auf den Haupt-Canvas ein
    draw.bitmap((x, y), im, fill=fill)

while True:
    with canvas(device) as draw:
        for block in range(16):
            block_group = block // 4      # 0,1,2,3 für jedes 4er-Modul
            pos_in_group = block % 4      # 0...3 innerhalb des Moduls

            if block_group % 2 == 0:
                # GERADE 4er-Module, Text normal (links -> rechts)
                x = 2 + 8 * block
                char = lst[(index + block) % len(lst)]
                text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
            else:
                # UNGERADE (= Zweites/Viertes) 4er-Modul, Text invertiert/gedreht
                block_start = 8 * (block_group * 4)
                x = block_start + (8 * (3 - pos_in_group)) + 2
                char = lst[(index + block) % len(lst)]
                draw_rotated_text(draw, x, 0, char, "white", proportional(LCD_FONT))
    time.sleep(0.5)
    index += 1
