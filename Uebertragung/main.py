import time
from PIL import Image, ImageDraw, ImageFont
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

# ===== KONFIGURATION =====
COLS = 16  # Anzahl Matrizen pro Zeile (horizontal)
ROWS = 2  # Anzahl Zeilen (vertikal)
NUM_MATRICES = COLS * ROWS

SCROLL_SPEED = 0.3  # Sekunden zwischen Updates
BRIGHTNESS = 10  # 0-255

# ===== SETUP =====
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=NUM_MATRICES, block_orientation=-90)
device.contrast(BRIGHTNESS)

# Text der durchlaufen soll
text_string = "The quick brown fox jumps over the lazy dog while curious developers eagerly explore advanced language features.    "

# Font laden
pil_font = ImageFont.load_default()


def draw_char_normal(draw, x, y, char):
    """Zeichen normal zeichnen (für Zeilen von links nach rechts)"""
    draw.text((x, y), char, font=pil_font, fill="white")


def draw_char_rotated(draw, x, y, char):
    """Zeichen 180° gedreht zeichnen (für Zeilen von rechts nach links)"""
    temp_img = Image.new("1", (8, 8))
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((2, -1), char, font=pil_font, fill=1)
    temp_img = temp_img.rotate(180)
    draw.bitmap((x, y), temp_img, fill="white")


def get_matrix_position(col, row):
    """
    Berechnet die tatsächliche Matrix-Nummer basierend auf Serpentine-Layout

    col: Spalte (0 bis COLS-1)
    row: Zeile (0 bis ROWS-1)

    Returns: Matrix-Index im CASCADE
    """
    if row % 2 == 0:
        # Gerade Zeilen: von links nach rechts
        return row * COLS + col
    else:
        # Ungerade Zeilen: von rechts nach links
        return row * COLS + (COLS - 1 - col)


scroll_offset = 0

while True:
    with canvas(device) as draw:
        char_index = 0

        for row in range(ROWS):
            for col in range(COLS):
                # Welches Zeichen soll an dieser Position angezeigt werden?
                current_char = text_string[(char_index + scroll_offset) % len(text_string)]

                # Matrix-Position berechnen
                matrix_num = get_matrix_position(col, row)

                # Pixel-Position auf dem Gesamt-Display
                # Jede Matrix ist 8x8 Pixel
                x_pixel = matrix_num * 8 + 2  # +2 für leichte Zentrierung
                y_pixel = 0

                # Zeichen zeichnen (normal oder gedreht je nach Zeile)
                if row % 2 == 0:
                    draw_char_normal(draw, x_pixel, y_pixel, current_char)
                else:
                    draw_char_rotated(draw, x_pixel, y_pixel, current_char)

                char_index += 1

    scroll_offset += 1
    time.sleep(SCROLL_SPEED)