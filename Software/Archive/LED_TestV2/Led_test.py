import spidev
import time

# open SPI bus 0, device 0 (CS0)
spi = spidev.SpiDev()
spi.open(10, 0)
spi.max_speed_hz = 1000000  # 1 MHz

def send_cmd(register, data):
    # Send the same command to all 4 cascaded modules
    spi.xfer2([register, data] * 4)

def init_max7219():
    send_cmd(0x09, 0x00)  # no decode
    send_cmd(0x0A, 0x0F)  # intensity = max
    send_cmd(0x0B, 0x07)  # scan limit = all 8 digits
    send_cmd(0x0C, 0x01)  # normal operation (not shutdown)
    send_cmd(0x0F, 0x00)  # display test off
    send_cmd(0x0F, 0x01)  # turn on display test
    time.sleep(2)
    send_cmd(0x0F, 0x00)  # back to normal mode

    clear_display()

def clear_display():
    for i in range(1, 9):
        send_cmd(i, 0x00)

def all_on():
    for i in range(1, 9):
        send_cmd(i, 0xFF)

init_max7219()

print("Lighting up all LEDs...")
all_on()
time.sleep(5)

print("Clearing...")
clear_display()

spi.close()