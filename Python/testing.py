import spidev
spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000
print("Sending test bytes...")
spi.xfer2([0xAA, 0x55, 0xFF, 0x00])
print("Done.")
spi.close()
