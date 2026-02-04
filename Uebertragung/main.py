import time
import math
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

# ===== KONFIGURATION =====
COLS = 16  # Anzahl Matrizen pro Zeile (horizontal)
ROWS = 2  # Anzahl Zeilen (vertikal)
NUM_MATRICES = COLS * ROWS

SNAKE_LENGTH = 40  # Länge der Schlange in Pixeln
SNAKE_SPEED = 0.05  # Sekunden zwischen Updates
BRIGHTNESS = 5

# ===== SETUP =====
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=NUM_MATRICES, block_orientation=90)
device.contrast(BRIGHTNESS)


def get_serpentine_path():
    """
    Erstellt den Pfad durch alle LEDs im Serpentine-Muster
    Returns: Liste von (x, y) Koordinaten
    """
    path = []

    for row in range(ROWS):
        if row % 2 == 0:
            # Gerade Zeilen: links nach rechts
            for col in range(COLS):
                for x in range(8):
                    matrix_num = row * COLS + col
                    pixel_x = matrix_num * 8 + x
                    for y in range(8):
                        path.append((pixel_x, y))
        else:
            # Ungerade Zeilen: rechts nach links
            for col in range(COLS - 1, -1, -1):
                for x in range(7, -1, -1):
                    matrix_num = row * COLS + col
                    pixel_x = matrix_num * 8 + x
                    for y in range(8):
                        path.append((pixel_x, y))

    return path


# Pfad generieren
serpentine_path = get_serpentine_path()
total_pixels = len(serpentine_path)

offset = 0

while True:
    with canvas(device) as draw:
        # Anzahl Schlangen berechnen die Platz haben
        num_snakes = max(1, total_pixels // (SNAKE_LENGTH * 2))

        for snake_num in range(num_snakes):
            # Startposition für diese Schlange
            snake_offset = (offset + snake_num * (total_pixels // num_snakes)) % total_pixels

            # Schlange zeichnen
            for i in range(SNAKE_LENGTH):
                position_in_path = (snake_offset + i) % total_pixels
                x, y = serpentine_path[position_in_path]

                # Helligkeit basierend auf Position in der Schlange
                # Kopf (i=0) = hell, Ende (i=SNAKE_LENGTH-1) = dunkel
                brightness = int(255 * (1 - i / SNAKE_LENGTH))

                # Zeichne Pixel mit unterschiedlicher Helligkeit
                if brightness > 30:  # Nur sichtbare Pixel zeichnen
                    color = "white" if brightness > 200 else "white"
                    draw.point((x, y), fill=color)

    offset = (offset + 1) % total_pixels
    time.sleep(SNAKE_SPEED)