import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import ceil
import pandas as pd
import pickle
from utils import generator


def train(training_data: list, model, model_name: str):
    full = []
    for td in training_data:
        with open(td, 'rb') as file:
            full += pickle.load(file)
    track1_df = pd.DataFrame([(i, a[0]) for i, a in full], columns=['Image', 'Angle'])
    train_samples, valid_samples = train_test_split(track1_df, test_size=0.2)
    train_generator = generator(train_samples)
    valid_generator = generator(valid_samples)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)
    cp = ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(train_generator,
                        steps_per_epoch=ceil(len(train_samples)/32),
                        validation_data=valid_generator,
                        validation_steps=ceil(len(valid_samples)/32),
                        epochs=60, callbacks=[es, cp])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(model_name + '-plot')


def train_full(training_data: list, model, model_name: str):
    full = []
    for td in training_data:
        with open(td, 'rb') as file:
            full += pickle.load(file)
    track1_df = pd.DataFrame(full, columns=['Image', 'Angle'])
    train_samples, valid_samples = train_test_split(track1_df, test_size=0.2)
    train_generator = generator(train_samples)
    valid_generator = generator(valid_samples)
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)
    cp = ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(train_generator,
                        steps_per_epoch=ceil(len(train_samples)/32),
                        validation_data=valid_generator,
                        validation_steps=ceil(len(valid_samples)/32),
                        epochs=60, callbacks=[es, cp])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(model_name + '-plot')
