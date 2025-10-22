import spidev, time

spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000

spi.xfer2([0x9, 0])
spi.xfer2([0xA, 0x7])
spi.xfer2([0xB, 0x0])
spi.xfer2([0xC, 1])
spi.xfer2([0xF, 1])
time.sleep(3)
spi.xfer2([0xC, 0])
spi.close()
