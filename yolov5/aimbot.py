#================================================================
#
#   File name   : YOLO_aimbot_main.py
#   Author      : PyLessons
#   Created date: 2020-10-06
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : CSGO main yolo aimbot script
#
#================================================================
import os
from tkinter import W

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import time
from datetime import datetime
import cv2
import mss
import numpy as np
import tensorflow as tf
import win32gui
import random
from prediction import predict
mx= 0
my= 0
mw = 0
mh = 0
def getWindowsCoords():
    coords = []
    def winEnumHandler( hwnd, ctx ):
        if win32gui.IsWindowVisible( hwnd ):
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1]
            w = rect[2] - x
            h = rect[3] - y
            print (hex(hwnd), win32gui.GetWindowText( hwnd ))

            if win32gui.GetWindowText(hwnd) == "Condition Zero":
                print("found")
                
                coords.append(x)
                coords.append(y)
                coords.append(w)
                coords.append(h)

    win32gui.EnumWindows( winEnumHandler, coords)
    return coords



coords = getWindowsCoords() # get window coords
print(coords)
x,y,w,h = coords
sct = mss.mss()
# pyautogui settings
import pyautogui # https://totalcsgo.com/commands/mouse
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0


offset = 30
times = []
sct = mss.mss()

print(x,y,w,h)
while True:
    t1 = time.time()
    img = np.array(sct.grab({"top": y-30, "left": x, "width": w, "height": h, "mon": -1}))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cwd = os.getcwd()
    print("current working dir: ", cwd)

    predict(weights='yolov5/runs/train/yolov5s_results8/weights/best.pt', source=img, view_img=True)
    # image, detection_list, bboxes = detect_enemy(yolo, np.copy(img), input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    # cv2.circle(image,(int(w/2),int(h/2)), 3, (255,255,255), -1) # center of weapon sight
    # cv2.imshow("OpenCV/Numpy normal", image)
    # th_list, t_list = [], []
    # for detection in detection_list:
    #     diff_x = (int(w/2) - int(detection[1]))*-1
    #     diff_y = (int(h/2) - int(detection[2]))*-1
    #     if detection[0] == "th":
    #         th_list += [diff_x, diff_y]
    #     elif detection[0] == "t":
    #         t_list += [diff_x, diff_y]

    # if len(th_list)>0:
    #     new = min(th_list[::2], key=abs)
    #     index = th_list.index(new)
    #     pyautogui.move(th_list[index], th_list[index+1])
    #     if abs(th_list[index])<12:
    #         pyautogui.click()
    # elif len(t_list)>0:
    #     new = min(t_list[::2], key=abs)
    #     index = t_list.index(new)
    #     pyautogui.move(t_list[index], t_list[index+1])
    #     if abs(t_list[index])<12:
    #         pyautogui.click()

    # t2 = time.time()
    # times.append(t2-t1)
    # times = times[-50:]
    # ms = sum(times)/len(times)*1000
    # fps = 1000 / ms
    # print("FPS", fps)
    # image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    
  
    # #if cv2.waitKey(25) & 0xFF == ord("q"):
    #     #cv2.destroyAllWindows()
    #     #break