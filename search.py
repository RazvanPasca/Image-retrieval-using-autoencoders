import os
from functools import reduce

import cv2
import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


def get_image_database():
    images = x_train.reshape((-1, 32, 32, 3))
    return encoder.predict(images)


def load_image(path):
    image = cv2.imread(os.path.join(os.curdir, path))
    image = cv2.resize(image, (32, 32))
    return image.astype('float32') / 255.


def get_random_image():
    indexes = np.arange(0, x_test.shape[0], 1)
    image = x_test[100]
    return image


def get_distance(image1, image2, type="MSE"):
    size = reduce((lambda x, y: x * y), image1.shape)
    if type is "MSE":
        return np.sum((image1 - image2) ** 2) / float(size)
    elif type is "L2":
        return np.linalg.norm(image1 - image2)
    elif type is "COS":
        return np.dot(image1, image2) / (np.linalg.norm(image1) * np.linalg.norm(image2))


def search_in_embeddings(input_image, nr_matches):
    image_database = get_image_database()
    enc_image = encoder.predict(input_image.reshape(-1, 32, 32, 3))
    distance_to_images = np.array([get_distance(enc_image, image) for image in image_database])
    distances_with_images = np.stack((distance_to_images, np.arange(0, x_train.shape[0], 1)), axis=-1)
    top_sorted_distances = distances_with_images[distances_with_images[:, 0].argsort()[:nr_matches]]
    top_images_index = top_sorted_distances[:, -1].astype(int)
    print("Embeddings top", top_images_index)
    return top_images_index


def search_in_orig_images(input_image, nr_matches):
    distance_to_images = np.array([get_distance(input_image, image) for image in x_train])
    distances_with_images = np.stack((distance_to_images, np.arange(0, x_train.shape[0], 1)), axis=-1)
    top_sorted_distances = distances_with_images[distances_with_images[:, 0].argsort()[:nr_matches]]
    top_images_index = top_sorted_distances[:, -1].astype(int)
    print("Originals top", top_images_index)
    return top_images_index


def resize_images(images, size=280):
    return np.array([cv2.resize(image, (size, size)) for image in images])


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


# Load previsouly trained model
model_name = 'cifar/autoencoder_conv_mse.h5'
autoencoder = load_model(model_name)
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)


def main():
    query_img = get_random_image()
    resized_img = cv2.resize(query_img, (280, 280))
    cv2.imshow('query img resized', resized_img)
    cv2.waitKey()

    results_index = search_in_embeddings(query_img, 9)
    images_to_plot = x_train[results_index]
    images_plot = gallery(resize_images(images_to_plot))
    plt.title("Results by embeddings {}".format(model_name))
    plt.imshow(images_plot)
    plt.show()

    results_index = search_in_orig_images(query_img, 9)
    images_to_plot = x_train[results_index]
    images_plot = gallery(resize_images(images_to_plot))
    plt.title("Results by originals")
    plt.imshow(images_plot)
    # plt.show()
    cv2.waitKey()


if __name__ == '__main__':
    main()
