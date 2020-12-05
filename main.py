import driving
from windows import Window


if __name__ == '__main__':
    window = Window('Grand Theft Auto V', 310, 26, 2)
    driving.drive(window, 'xception.h5', 5)
