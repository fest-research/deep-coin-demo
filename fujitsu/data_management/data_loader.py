import os
import numpy as np
from scipy.ndimage import imread
from keras.utils import np_utils

from fujitsu.utils.log import setup_logger
log = setup_logger("data_loader")


def load_dataset(data_dir):
    # set the numpy random seed for reproducibility
    np.random.seed(42)

    # load all of the data
    samples, labels, mean = assemble_data_set(data_dir)
    n = labels.shape[0]
    log.info("Loaded data with total size: {}".format(n))

    # split into a training and test set
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_n = int(0.7 * n)
    indices_train = indices[:train_n]
    indices_test = indices[train_n:]
    dataset = {
        'X_train': samples[indices_train],
        'y_train': labels[indices_train],
        'X_test': samples[indices_test],
        'y_test': labels[indices_test]
    }
    return dataset, mean


def assemble_data_set(data_dir):
    xs = list()
    ys = list()
    for fn in os.listdir(os.path.join(data_dir, "positive")):
        x = imread(os.path.join(data_dir, "positive", fn))
        xs.append(x.transpose((2, 0, 1)))
        ys.append(1)

    for fn in os.listdir(os.path.join(data_dir, "negative")):
        x = imread(os.path.join(data_dir, "negative", fn))
        xs.append(x.transpose((2, 0, 1)))
        ys.append(0)

    # make the labels one-hot encoded
    ys = np_utils.to_categorical(np.array(ys), nb_classes=2)
    xs = np.array(xs, dtype='float32')

    xs /= 255
    mean = np.mean(xs, axis=0)
    xs -= mean
    return xs, ys, mean


def load_mnist():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # some basic preprocessing
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == "__main__":
    # test the loading
    load_dataset(os.path.join(os.path.dirname(__file__), "../../data"))
