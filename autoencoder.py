import numpy as np
from keras import losses
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, regularizers
from keras.models import Model, load_model
from keras.optimizers import Adam, Adadelta

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (-1, 32, 32, 3))
x_test = np.reshape(x_test, (-1, 32, 32, 3))


def create_encoder(activation='relu', padding='same', name='encoder'):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(16, (3, 3), activation=activation, padding=padding)(input_img)
    x = MaxPooling2D((2, 2), padding=padding)(x)

    x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
    x = MaxPooling2D((2, 2), padding=padding)(x)

    x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
    x = MaxPooling2D((2, 2), padding=padding)(x)

    x = Conv2D(128, (3, 3), activation=activation, padding=padding)(x)
    x = MaxPooling2D((2, 2), padding=padding)(x)

    x = Flatten()(x)
    encoded = Dense(200, activation=activation, name=name)(x)
    return input_img, encoded


def create_decoder(input, activation='relu', padding='same', name='decoder'):
    x = Dense(2 * 2 * 128)(input)
    x = Reshape((2, 2, 128))(x)

    x = Conv2D(128, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation=activation, padding=padding)(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name=name)(x)
    return decoded


def train_model(input_img, decoder, load=False):
    if load:
        print('loading model')
        autoencoder = load_model('cifar/autoencoder_conv_mse.h5')
    else:
        autoencoder = Model(input_img, decoder)

    optimizer = Adam(lr=0.002, clipnorm=.5)
    autoencoder.compile(optimizer=optimizer, loss=losses.mean_squared_error)
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/cifar_dense_mse_large', histogram_freq=10, write_graph=True)])
    autoencoder.save('cifar/autoencoder_dense_mse_large.h5')


def main():
    input_img, encoder = create_encoder()
    decoder = create_decoder(encoder)
    train_model(input_img, decoder)


if __name__ == '__main__':
    main()
