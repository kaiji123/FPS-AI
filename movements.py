from pyautogui import sleep
from directkeys import * 
import time
start = time.time()
while True:
    sleep(5)
    PressKey(W)
    sleep(1)
    if time.time() - start > 1000:
        break

    