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


def get_image_database(encoder):
    images = x_train.reshape((-1, 32, 32, 3))
    return encoder.predict(images)


def get_val_database(encoder):
    images = x_test.reshape((-1, 32, 32, 3))
    return encoder.predict(images)


def load_image(path):
    image = cv2.imread(os.path.join(os.curdir, path))
    image = cv2.resize(image, (32, 32))
    return image.astype('float32') / 255.


def resize_images(images, size=280):
    return np.array([cv2.resize(image, (size, size)) for image in images])


def get_single_accuracy(target_index, top_images_index, rank):
    true_positives = y_test[target_index] == y_train[top_images_index[:rank]]
    return true_positives


def save_results(*accuracies, model_name, metric, path):
    accuracies = list(accuracies)
    with open(path, 'a') as results_file:
        results_file.write(
            "Model name:{} Metric: {} top 5 accuracy:{} top 10 accuracy:{}\n".format(model_name, metric, accuracies[0],
                                                                                     accuracies[1]))


def get_distance(image1, image2, metric):
    size = reduce((lambda x, y: x * y), image1.shape)

    if metric is "L2":
        return np.linalg.norm(image1 - image2)
    elif metric is "COS":
        image1_flatten = image1.flatten()
        image2_flatten = image2.flatten()
        # Want the negative because I order them in descending order
        return -np.dot(image1_flatten, image2_flatten) / (
                np.linalg.norm(image1_flatten) * np.linalg.norm(image2_flatten))


def search_in_embeddings(input_index, nr_matches, encoder, metric):
    image_database = get_image_database(encoder)
    enc_images = get_val_database(encoder)
    enc_image = enc_images[input_index]

    distance_to_images = np.array([get_distance(enc_image, image, metric) for image in image_database])
    distances_with_images = np.stack((distance_to_images, np.arange(0, x_train.shape[0], 1)), axis=-1)
    top_sorted_distances = distances_with_images[distances_with_images[:, 0].argsort()[:nr_matches]]
    top_images_index = top_sorted_distances[:, -1].astype(int)
    return top_images_index


def search_in_orig_images(input_index, nr_matches, metric):
    input_image = x_test[input_index]
    distance_to_images = np.array([get_distance(input_image, image, metric) for image in x_train])
    distances_with_images = np.stack((distance_to_images, np.arange(0, x_train.shape[0], 1)), axis=-1)
    top_sorted_distances = distances_with_images[distances_with_images[:, 0].argsort()[:nr_matches]]
    top_images_index = top_sorted_distances[:, -1].astype(int)
    return top_images_index


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def process_image(encoder, index, model_name, nr_matches, metric="MSE", use_embeddings=True):
    if use_embeddings:
        results_index = search_in_embeddings(index, nr_matches,
                                             encoder, metric)
    else:
        results_index = search_in_orig_images(index,
                                              nr_matches, metric)
    # images_to_plot = x_train[results_index]
    # images_plot = gallery(resize_images(images_to_plot))
    # plt.title("Results by {} with metric {}".format(model_name, metric))
    # plt.imshow(images_plot)
    # plt.show()
    return get_single_accuracy(index, results_index, 5), get_single_accuracy(index, results_index, 10)


def main():
    # Load previsouly trained model
    model_name = 'cifar/autoencoder_dense_binary.h5'
    autoencoder = load_model(model_name)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

    path = "results.txt"
    nr_matches = 10
    showed = False
    batch_size = 500

    indexes = np.random.choice(x_test.shape[0], batch_size)

    for metric in ["L2", "COS"]:

        accuracy_5_embed = 0
        accuracy_10_embed = 0
        accuracy_5_orig = 0
        accuracy_10_orig = 0

        for image_index in indexes:
            query_img = x_test[image_index]
            # if not showed:
            #     resized_img = cv2.resize(query_img, (280, 280))
            #     plt.title('Query img resized')
            #     plt.imshow(resized_img)
            #     plt.show()
            #     # showed = True

            accuracy_5_embed_iter, accuracy_10_embed_iter = process_image(encoder, image_index, model_name, nr_matches,
                                                                          metric)
            accuracy_5_embed += accuracy_5_embed_iter
            accuracy_10_embed += accuracy_10_embed_iter

            accuracy_5_orig_iter, accuracy_10_orig_iter = process_image(encoder, image_index, "Original images",
                                                                        nr_matches,
                                                                        metric,
                                                                        use_embeddings=False)
            accuracy_5_orig += accuracy_5_orig_iter
            accuracy_10_orig += accuracy_10_orig_iter

        accuracy_5_embed = np.sum(accuracy_5_embed) / (5 * batch_size)
        accuracy_10_embed = np.sum(accuracy_10_embed) / (10 * batch_size)
        accuracy_5_orig = np.sum(accuracy_5_orig) / (5 * batch_size)
        accuracy_10_orig = np.sum(accuracy_10_orig) / (10 * batch_size)

        save_results(accuracy_5_embed, accuracy_10_embed, metric=metric, model_name=model_name, path=path)
        save_results(accuracy_5_orig, accuracy_10_orig, metric=metric, model_name="Original images", path=path)


if __name__ == '__main__':
    main()
