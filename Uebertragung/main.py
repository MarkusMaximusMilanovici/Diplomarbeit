from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

# Initialisierung des SPI Interface für die LED Matrix (ein Modul/Display)
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, width=8, height=8, block_orientation=0)

# Beispiel: Einzelner Pixel einschalten (x=3, y=5)
with canvas(device) as draw:
    draw.point((3, 5), fill="white")

# Beispiel: Muster anzeigen
with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white", fill="black")
