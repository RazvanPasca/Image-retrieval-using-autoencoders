import os

import keras as K
import numpy as np
from keras import losses
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (-1, 32, 32, 3))
x_test = np.reshape(x_test, (-1, 32, 32, 3))
y_train = K.utils.to_categorical(y_train, 10)
y_test = K.utils.to_categorical(y_test, 10)


def create_encoder(activation='relu', padding='same', name='encoder'):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(16, (3, 3), activation=activation, padding=padding)(input_img)
    x = MaxPooling2D((2, 2), padding=padding)(x)
    # x= Dropout(0.25)(x)
    # 16x16
    x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
    x = MaxPooling2D((2, 2), padding=padding)(x)

    # 8x8
    x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
    x = MaxPooling2D((2, 2), padding=padding)(x)
    #  x= Dropout(0.25)(x)
    # 4x4
    # x = Conv2D(96, (3, 3), activation=activation, padding=padding)(x)
    # x = MaxPooling2D((2, 2), padding=padding)(x)
    # 2x2
    x = Flatten(name=name)(x)
    return input_img, x


def create_decoder(input, activation='relu', padding='same', name='decoder'):
    x = Dense(4 * 4 * 64)(input)
    x = Reshape((4, 4, 64))(x)

    # x = Conv2D(96, (3, 3), activation=activation, padding=padding)(x)
    # x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation=activation, padding=padding)(x)

    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name=name)(x)

    logits = Dense(64, activation=activation)(input)
    logits = Dropout(0.25)(logits)
    predicted_classes = Dense(10, activation='softmax', name="classifier")(logits)

    return decoder_output, predicted_classes


name = 'autoencoder_3lay_sparse_sigmoid_class_4v1_100e.h5'


def train_model(input_img, decoder, classifier, load=False):
    if load:
        print('loading model')
        autoencoder = load_model('models/' + name)
    else:
        autoencoder = Model(input_img, outputs=[decoder, classifier])

    optimizer = Adam(lr=0.002, clipnorm=.5)
    autoencoder.compile(optimizer=optimizer, loss=[losses.binary_crossentropy, losses.categorical_crossentropy],
                        loss_weights=[4.0, 1.0])
    autoencoder.fit(x_train, [x_train, y_train],
                    epochs=100,
                    batch_size=32,
                    verbose=2,
                    shuffle=True,
                    validation_data=(x_test, [x_test, y_test]),
                    callbacks=[TensorBoard(log_dir='tboard/' + name, histogram_freq=10, write_graph=True)])
    autoencoder.save('models/' + name)


def main():
    input_img, encoder = create_encoder()
    decoder, classifier = create_decoder(encoder)
    train_model(input_img, decoder, classifier)


if __name__ == '__main__':
    main()
