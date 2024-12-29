import json
import pyboy
import os
from pyboy.utils import WindowEvent

valid_actions = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
        WindowEvent.PRESS_BUTTON_START,
    ]

release_actions = [
        WindowEvent.RELEASE_ARROW_DOWN,
        WindowEvent.RELEASE_ARROW_LEFT,
        WindowEvent.RELEASE_ARROW_RIGHT,
        WindowEvent.RELEASE_ARROW_UP,
        WindowEvent.RELEASE_BUTTON_A,
        WindowEvent.RELEASE_BUTTON_B,
        WindowEvent.RELEASE_BUTTON_START
    ]


def run_action_on_emulator(emulator, action):
    # press button then release after some steps
    emulator.send_input(valid_actions[action])
    press_step = 8
    emulator.tick(press_step, True)
    emulator.send_input(release_actions[action])
    emulator.tick(24 - press_step - 1, True)
    emulator.tick(1, True)


if __name__ == "__main__":
    

    emulator = pyboy.PyBoy(
            os.path.expanduser("~/rom/PokemonRed.gb"),
            window="SDL2",
        )
    with open(os.path.expanduser("~/rom/PokemonRed.gb.state"), "rb") as f:
        emulator.load_state(f)

    sequence = os.path.join(os.path.dirname(__file__), "best.json")
    with open(sequence, "r") as f:
        data = json.load(f)
        
    for i in data["action_sequence"]:
        run_action_on_emulator(emulator, i)
    
    
