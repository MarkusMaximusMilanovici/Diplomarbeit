import spidev, time

spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

spi.xfer2([0x9, 0])
spi.xfer2([0xA, 0x7])
spi.xfer2([0xB, 0xFF])
spi.xfer2([0x1, 0xFF])
time.sleep(3)
spi.xfer2([0xC, 0])
spi.close()

def send_cmd(reg, data):
    spi.xfer2([reg, data] * 4)

def init():
    send_cmd(0x09, 0x00)
    send_cmd(0x0A, 0x0F)
    send_cmd(0x0B, 0x07)
    send_cmd(0x0C, 0x01)
    send_cmd(0x0F, 0x01)  # TEST MODE ON
    time.sleep(3)
    send_cmd(0x0F, 0x00)  # TEST MODE
