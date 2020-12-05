import sklearn as skl
import numpy as np
import cv2 as cv


def random_shift(img, steering, shift_range=20):
    ht, wd, ch = img.shape

    shift_x = shift_range * (np.random.rand() - 0.5)
    shift_y = shift_range * (np.random.rand() - 0.5)
    shift_m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv.warpAffine(img, shift_m, (wd, ht))

    steering += shift_x * 0.002
    return img, steering


def random_shadow(img):
    ht, wd, ch = img.shape
    x1, y1 = wd * np.random.rand(), 0
    x2, y2 = wd * np.random.rand(), ht
    xm, ym = np.mgrid[0:ht, 0:wd]

    mask = np.zeros_like(img[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.6, high=0.9)

    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv.cvtColor(hls, cv.COLOR_HLS2RGB)


def adjust_brightness(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * ratio
    return cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)


def flip(img, steering):
    if np.random.rand() < 0.5:
        img = cv.flip(img, 1)
        if steering != 0:
            steering = -steering
    return img, steering


def augment_img(img, steering):
    img, steering = flip(img, steering)
    img, steering = random_shift(img, steering)
    img = random_shadow(img)
    img = adjust_brightness(img)
    return img, steering


def get_train_test_labels(df):
    images = []
    steering_angle_list = []
    for index, row in df.iterrows():
        # angle = float(row['Angle'])
        angle = row['Angle']
        cimg = row['Image']
        # cimg, angle = augment_img(cimg, angle)
        images.append(cimg)
        steering_angle_list.append(angle)
    return images, steering_angle_list


def generator(df, bs=32):
    total = len(df)
    while 1:
        skl.utils.shuffle(df)
        for offset in range(0, total, bs):
            batch = df[offset:offset + bs]
            images, angles = get_train_test_labels(batch)
            x_train = np.array(images)
            y_train = np.array(angles)
            yield tuple(skl.utils.shuffle(x_train, y_train))
