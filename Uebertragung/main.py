import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=16, block_orientation=-90)  # 16 Module in Serie

device.contrast(10)
lst = "Das Crazy euda wir fahrn zu WM "
index = 0

# Funktion: Buchstaben vertikal zeichnen (wie vertical_text)
def draw_vertical(draw, x, y, char, fill, font):
    for i, c in enumerate(char):
        text(draw, (x, y + i * 8), c, fill=fill, font=proportional(LCD_FONT))

while True:
    with canvas(device) as draw:
        for block in range(16):
            x = 2 + 8 * block   # x-Position für jeden Block mittig
            char = lst[(index + block) % len(lst)]
            # Alle 4 Blöcke umschalten: 0–3, 8–11 horizontal; 4–7, 12–15 vertikal
            if (block // 4) % 2 == 0:  # Gruppenweise (erste + dritte 4er-Blöcke)
                text(draw, (x, 1), char, fill="white", font=proportional(LCD_FONT))
            else:                      # zweite + vierte 4er-Blöcke vertikal
                draw_vertical(draw, x, 0, char, "white", proportional(LCD_FONT))
    time.sleep(0.5)
    index += 1
