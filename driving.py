import keras
from keras.models import Sequential
from windows import Window
from input import grab_screen
from output import JoystickEmulator


def drive(window: Window, model_path: str, multiplier: int):
    model: Sequential = keras.models.load_model(model_path)
    window.align()
    jd = JoystickEmulator()
    while True:
        angle = multiplier * float(model.predict(grab_screen(window)[None, :, :, :], batch_size=1))
        jd.emulate(angle)
        print(angle)


def drive_full(window: Window, model_path: str, angle_multiplier: int, acce_multiplier: int):
    model: Sequential = keras.models.load_model(model_path)
    window.align()
    jd = JoystickEmulator()
    while True:
        prediction = model.predict(grab_screen(window)[None, :, :, :], batch_size=1)[0]
        angle = angle_multiplier * float(prediction[0])
        acce = acce_multiplier * float(prediction[1])
        jd.emulate(angle)
        print(angle, acce)
