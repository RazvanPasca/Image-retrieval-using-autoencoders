import cv2
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
from search import gallery, resize_images
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

nr_images = 9


def main():
    print('Loading model :')
    # Load previously trained autoencoder
    autoencoder = load_model('cifar/autoencoder_conv_mse.h5')
    images_index = np.random.choice(x_test.shape[0], nr_images)

    to_predict = x_test[images_index]
    plt.imshow(gallery(resize_images(to_predict, 280)))
    plt.show()

    images = autoencoder.predict(to_predict.reshape(nr_images, x_test.shape[1], x_test.shape[2], 3))
    results = gallery(resize_images(images, 280))
    plt.imshow(results)
    plt.show()


if __name__ == '__main__':
    main()
