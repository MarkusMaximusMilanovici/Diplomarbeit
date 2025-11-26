import time
import numpy as np
from PIL import Image
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
import vizdoom as vzd

# LED Matrix Setup
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, block_orientation=-90, width=32, height=32)
device.contrast(10)

print(f"Display: {device.width}x{device.height}")

# ViZDoom Setup
game = vzd.DoomGame()

# Basis-Config
game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
game.set_doom_map("map01")

# Screen auf niedrigste Auflösung für Performance
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_screen_format(vzd.ScreenFormat.GRAY8)

# Grafik reduzieren für bessere Performance
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(False)  # Waffe ausblenden = schneller
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)

# Kein Fenster auf Pi (headless)
game.set_window_visible(False)  # Spart Performance!

# Buttons
game.set_available_buttons([
    vzd.Button.MOVE_FORWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK
])

game.set_mode(vzd.Mode.PLAYER)
game.set_ticrate(35)

game.init()
print("Doom läuft auf dem Pi!")

# Einfache Demo: Auto-Play
episodes = 1

for episode in range(episodes):
    game.new_episode()

    step = 0
    while not game.is_episode_finished():
        state = game.get_state()

        if state:
            # Screen holen und skalieren
            screen = state.screen_buffer
            img = Image.fromarray(screen)
            img_resized = img.resize((32, 32), Image.Resampling.NEAREST)
            frame = np.array(img_resized)

            # Deine Block-Rotation
            zweite = frame[8:16, :]
            vierte = frame[24:32, :]
            frame[8:16, :] = np.rot90(zweite, 2)
            frame[24:32, :] = np.rot90(vierte, 2)

            # LED anzeigen
            display_img = Image.fromarray(frame)
            display_img = display_img.convert(device.mode)
            device.display(display_img)

        # Einfache KI: vorwärts laufen + gelegentlich drehen
        action = [0, 0, 0, 0]  # [vorwärts, links, rechts, schießen]

        if step % 20 == 0:  # alle 20 Frames drehen
            action[1] = 1  # links
        else:
            action[0] = 1  # vorwärts

        game.make_action(action)
        step += 1

        time.sleep(1 / 35)  # 35 FPS

    print(f"Episode beendet. Punkte: {game.get_total_reward()}")

game.close()
