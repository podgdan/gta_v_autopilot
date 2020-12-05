import pyvjoy as vj
import cv2 as cv


class JoystickEmulator:

    def __init__(self):
        self.__instance = vj.VJoyDevice(1)

    def emulate(self, turn):
        if -1 <= turn <= 1:
            self.__instance.set_axis(vj.HID_USAGE_X, round(turn * 16384 + 16384))


def show_screen(img_gen):
    for img in img_gen:
        cv.imshow('output', img)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
