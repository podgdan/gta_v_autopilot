from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras.models import Model


DEFAULT_INPUT_SHAPE = (66, 200, 3)


def pilotnet(input_shape=DEFAULT_INPUT_SHAPE):

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(1e-4, decay=0.0))

    return model


def pilotnet_full(input_shape=DEFAULT_INPUT_SHAPE):

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2))

    model.compile(loss='mse', optimizer=Adam(1e-4, decay=0.0))

    return model


def xception(input_shape=DEFAULT_INPUT_SHAPE):
    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='mse',
                  optimizer=Adam(0.001, decay=1e-5),
                  metrics=['accuracy'])
    return model
