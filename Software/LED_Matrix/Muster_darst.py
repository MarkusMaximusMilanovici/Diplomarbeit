import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

# ===== KONFIGURATION =====
# 1 Modul = 4 einzelne 8x8 LED-Matrizen
COLS = 16
ROWS = 1
NUM_MATRICES = COLS * ROWS
BRIGHTNESS = 5

# ===== SETUP =====
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=NUM_MATRICES, block_orientation=-90)
device.contrast(BRIGHTNESS)

# Gesamtgröße für eine Reihe aus 4 Matrizen
WIDTH = NUM_MATRICES * 8
HEIGHT = 8


def pattern_all_on():
    """Alle LEDs an"""
    with canvas(device) as draw:
        draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), fill="white")


def pattern_checkerboard():
    """Schachbrettmuster"""
    with canvas(device) as draw:
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if (x + y) % 2 == 0:
                    draw.point((x, y), fill="white")


def pattern_vertical_lines():
    """Vertikale Linien"""
    with canvas(device) as draw:
        for x in range(0, WIDTH, 2):
            draw.line((x, 0, x, HEIGHT - 1), fill="white")


def pattern_horizontal_lines():
    """Horizontale Linien"""
    with canvas(device) as draw:
        for y in range(0, HEIGHT, 2):
            draw.line((0, y, WIDTH - 1, y), fill="white")


def pattern_diagonal():
    """Diagonale Linien"""
    with canvas(device) as draw:
        for x in range(WIDTH):
            y = x % HEIGHT
            draw.point((x, y), fill="white")


def pattern_wave(offset):
    """Wellenmuster"""
    import math
    with canvas(device) as draw:
        for x in range(WIDTH):
            y = int((HEIGHT - 1) / 2 + ((HEIGHT - 1) / 2) * math.sin((x + offset) * 0.3))
            draw.point((x, y), fill="white")


def pattern_matrix_borders():
    """Zeigt die Grenzen jeder 8x8 Matrix"""
    with canvas(device) as draw:
        for i in range(NUM_MATRICES):
            x = i * 8
            draw.rectangle((x, 0, x + 7, HEIGHT - 1), outline="white")


patterns = [
    ("Alle an", pattern_all_on),
    ("Schachbrett", pattern_checkerboard),
    ("Vertikale Linien", pattern_vertical_lines),
    ("Horizontale Linien", pattern_horizontal_lines),
    ("Diagonale", pattern_diagonal),
    ("Matrix Grenzen", pattern_matrix_borders),
]

print("Zeige verschiedene Muster...")

try:
    offset = 0

    while True:
        for name, pattern_func in patterns:
            print(f"Muster: {name}")
            pattern_func()
            time.sleep(2)

        print("Muster: Welle")
        for _ in range(30):
            pattern_wave(offset)
            offset += 1
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nProgramm beendet")
    device.clear()