import os

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model

from search import gallery, resize_images

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

nr_images = 12

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    name = 'autoencoder_conv_mse.h5'
    reconstr = 'reconstructions/'
    print('Loading model :')
    # Load previously trained autoencoder
    autoencoder = load_model('cifar/' + name)
    # images_index = np.random.choice(x_test.shape[0], nr_images)
    images_index = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    to_predict = x_test[images_index]
    orig_images = gallery(resize_images(to_predict, 280))
    fig = plt.imshow(orig_images)
    plt.savefig(reconstr + 'originals_val {}.{}'.format(name[:-3], 'png'))

    images = autoencoder.predict(to_predict.reshape(nr_images, x_test.shape[1], x_test.shape[2], 3))
    results = gallery(resize_images(images, 280))
    fig = plt.imshow(results)
    plt.savefig(reconstr + 'reconstruction_val {}.{}'.format(name[:-3], 'png'))

    to_predict = x_train[images_index]
    orig_images = gallery(resize_images(to_predict, 280))
    fig = plt.imshow(orig_images)
    plt.savefig(reconstr + 'originals_train {}.{}'.format(name[:-3], 'png'))

    images = autoencoder.predict(to_predict.reshape(nr_images, x_test.shape[1], x_test.shape[2], 3))
    results = gallery(resize_images(images, 280))
    fig = plt.imshow(results)
    plt.savefig(reconstr + 'reconstruction_train {}.{}'.format(name[:-3], 'png'))


if __name__ == '__main__':
    main()
