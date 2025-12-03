from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas
from PIL import Image
import time

# Initialisierung: 4 Blöcke = 4 Geräte in Kaskade
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=0, rotate=0)

# Helligkeit auf Maximum (0-255)
device.contrast(255)

def all_on():
    """Alle 16 LEDs (4 Blöcke) einschalten"""
    img = Image.new('1', (32, 8), color=1)  # 4 Blöcke à 8x8 = 32x8
    device.display(img)
    print("Alle LEDs AN")

def all_off():
    """Alle 16 LEDs (4 Blöcke) ausschalten"""
    device.clear()
    print("Alle LEDs AUS")

# Beispiel-Nutzung
if __name__ == "__main__":
    while True:
        all_on()
        time.sleep(2)
        all_off()
        time.sleep(2)
