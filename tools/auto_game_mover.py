"""
How to move the mouse and perform actions randomly using Python in Games for auto moves in Games?
"""

import time
import pyautogui
import random
from datetime import datetime
from pynput import mouse

last_mouse_move_time = time.time()
script_positions = []

def on_move(x, y):
    global last_mouse_move_time
    if (x, y) not in script_positions:
        last_mouse_move_time = time.time()

def click_randomly():
    try:
        # Set up the mouse listener
        listener = mouse.Listener(on_move=on_move)
        listener.start()

        while True:
            # Get the current time
            current_time = datetime.now().time()
            start_time = datetime.strptime("08:00", "%H:%M").time()
            end_time = datetime.strptime("19:00", "%H:%M").time()

            # Check if the current time is within the allowed range
            if not (start_time <= current_time <= end_time):
                time.sleep(300)
                continue

            # Check if there was recent human mouse movement
            if time.time() - last_mouse_move_time < 300:
                time.sleep(300)
                continue

            # Generate a random time to wait between 4 to 6 minutes (240 to 360 seconds)
            time_to_wait = random.uniform(240, 360)
            print(f"Waiting for {time_to_wait:.2f} seconds before the next action.")
            time.sleep(time_to_wait)

            # Check again before performing the action
            if time.time() - last_mouse_move_time < 300:
                time.sleep(300)
                continue

            # Move the mouse to the center of the screen and perform a right-click
            screen_width, screen_height = pyautogui.size()
            center_x, center_y = screen_width / 2, screen_height / 2
            script_positions.append((center_x, center_y))
            pyautogui.moveTo(center_x, center_y)
            pyautogui.click(button='right')
            print("Right-click performed at the center of the screen.")
            time.sleep(1)  # Short delay to ensure the move is processed
            script_positions.pop()  # Remove the position after action

            # Check again before performing the next action
            if time.time() - last_mouse_move_time < 300:
                print("Human activity detected. Pausing actions for 5 minutes.")
                time.sleep(300)
                continue

            # Move the mouse to a random position to the left or right of the screen and perform a Ctrl + click
            offset = random.randint(-screen_width // 4, screen_width // 4)
            target_x = center_x + offset
            script_positions.append((target_x, center_y))
            pyautogui.moveTo(target_x, center_y)
            pyautogui.keyDown('ctrl')
            pyautogui.click()
            pyautogui.keyUp('ctrl')
            print(f"Ctrl + click performed at position ({target_x}, {center_y}).")
            time.sleep(1)
            script_positions.pop()

    except KeyboardInterrupt:
        print("Script stopped by the user.")
    finally:
        listener.stop()

click_randomly()
