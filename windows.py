import win32gui as wg
import win32con as wc


class Window:

    def __init__(self, name: str, crop_top: int, crop_bot: int, shrink_by: int):
        self.hwnd = wg.FindWindow(None, name)
        wg.ShowWindow(self.hwnd, wc.SW_SHOWNORMAL)
        rect = wg.GetWindowRect(self.hwnd)
        self.x = -2
        self.y = 0
        self.width = rect[2] - rect[0]
        self.height = rect[3] - rect[1]
        self.capture_x = self.x + 5
        self.capture_y = self.y + 26
        self.capture_width = self.width - 6
        self.capture_height = self.height - 29
        self.cropped_y = self.capture_y + crop_top
        self.cropped_height = self.capture_height - crop_top - crop_bot
        self.target_width = int(self.capture_width / shrink_by)
        self.target_height = int(self.cropped_height / shrink_by)

    def align(self):
        wg.SetForegroundWindow(self.hwnd)
        wg.MoveWindow(self.hwnd, self.x, self.y, self.width, self.height, 1)
