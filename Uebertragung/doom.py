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
game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")  # oder deine eigene .wad
game.set_doom_map("map01")

# Screen auf 32x32 setzen für deine Matrix
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)  # wird auf 32x32 runterskaliert
game.set_screen_format(vzd.ScreenFormat.GRAY8)  # Graustufen
game.set_render_hud(False)  # HUD ausblenden für bessere Sicht
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)

# Fenster (optional, kann auch False sein)
game.set_window_visible(True)

# Verfügbare Aktionen definieren
game.set_available_buttons([
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK
])

# Game Mode
game.set_mode(vzd.Mode.PLAYER)  # Manuelles Spielen
game.set_ticrate(35)  # Doom Standard

# Game starten
game.init()

print("Doom läuft! Steuerung:")
print("  A/D - Links/Rechts drehen")
print("  W/S - Vor/Zurück")
print("  Q/E - Seitwärts")
print("  SPACE - Schießen")
print("  ESC - Beenden")

# Tasten-Mapping (mit keyboard library - optional)
try:
    import keyboard

    keyboard_available = True
except ImportError:
    keyboard_available = False
    print("Tipp: 'pip install keyboard' für Tastatur-Steuerung")
    print("Läuft jetzt mit zufälligen Aktionen...")

episodes = 10

for episode in range(episodes):
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()

        if state:
            # Screen Buffer holen (160x120 Graustufen)
            screen = state.screen_buffer

            # Auf 32x32 runterskalieren
            img = Image.fromarray(screen)
            img_resized = img.resize((32, 32), Image.Resampling.NEAREST)
            frame = np.array(img_resized)

            # Deine Display-Rotation anwenden (2. und 4. Block)
            zweite = frame[8:16, :]
            vierte = frame[24:32, :]
            frame[8:16, :] = np.rot90(zweite, 2)
            frame[24:32, :] = np.rot90(vierte, 2)

            # Auf LED-Matrix anzeigen
            display_img = Image.fromarray(frame)
            display_img = display_img.convert(device.mode)
            device.display(display_img)

        # Aktion basierend auf Tastatur (falls verfügbar)
        action = [0] * 7  # 7 Buttons

        if keyboard_available:
            if keyboard.is_pressed('a'):
                action[4] = 1  # TURN_LEFT
            if keyboard.is_pressed('d'):
                action[5] = 1  # TURN_RIGHT
            if keyboard.is_pressed('w'):
                action[2] = 1  # MOVE_FORWARD
            if keyboard.is_pressed('s'):
                action[3] = 1  # MOVE_BACKWARD
            if keyboard.is_pressed('q'):
                action[0] = 1  # MOVE_LEFT
            if keyboard.is_pressed('e'):
                action[1] = 1  # MOVE_RIGHT
            if keyboard.is_pressed('space'):
                action[6] = 1  # ATTACK
            if keyboard.is_pressed('esc'):
                break
        else:
            # Zufällige Aktion für Demo
            import random

            action[random.randint(0, 6)] = 1

        # Aktion ausführen
        game.make_action(action)
        time.sleep(1 / 35)  # 35 FPS (Doom Standard)

    print(f"Episode {episode + 1} beendet. Punkte: {game.get_total_reward()}")

game.close()
print("Doom beendet!")
