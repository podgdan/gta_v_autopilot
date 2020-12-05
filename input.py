import XInput as Xi
import win32gui as wg
import win32ui as wu
import win32con as wc
import numpy as np
import cv2 as cv
import pickle
from windows import Window


class JoystickDumper:

    def __init__(self):
        connected = Xi.get_connected()
        self.__user_index = next(i for i in range(len(connected)) if connected[i])

    def get_dump(self):
        state = Xi.get_state(self.__user_index)
        return Xi.get_thumb_values(state)[0][0], Xi.get_trigger_values(state)[1]


def grab_screen(window: Window):
    height = window.cropped_height
    dc = wg.GetWindowDC(window.hwnd)
    img_dc = wu.CreateDCFromHandle(dc)
    mem_dc = img_dc.CreateCompatibleDC()
    bmp = wu.CreateBitmap()
    bmp.CreateCompatibleBitmap(img_dc, window.capture_width, height)
    mem_dc.SelectObject(bmp)
    mem_dc.BitBlt((0, 0), (window.capture_width, height), img_dc, (window.capture_x, window.cropped_y), wc.SRCCOPY)
    arr = bmp.GetBitmapBits(True)
    img = np.fromstring(arr, dtype='uint8')
    img.shape = (height, window.capture_width, 4)
    img_dc.DeleteDC()
    mem_dc.DeleteDC()
    wg.ReleaseDC(window.hwnd, dc)
    wg.DeleteObject(bmp.GetHandle())
    return cv.resize(cv.cvtColor(img, cv.COLOR_RGBA2RGB), (window.target_width, window.target_height))


def record_dump(window: Window, size: int, path: str):
    jd = JoystickDumper()
    data = []
    for i in range(100):
        print(i)
        for j in range(int(size / 100)):
            data.append((grab_screen(window), jd.get_dump()))
    with open(path, 'wb') as output_file:
        pickle.dump(data, output_file)
