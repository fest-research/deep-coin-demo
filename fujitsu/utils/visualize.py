"""
Module for visualizations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize, imsave


def visualize_activations(model, test_sample, grid_shape):
    activations = model.activations(layer_index=2, samples=test_sample)[0][0]
    img_shape = activations.shape[1:]
    activations_image = create_tiles(activations, img_shape=img_shape,
                                     grid_shape=grid_shape,
                                     tile_spacing=(10, 10))
    activations_image = imresize(activations_image, size=(800, 800))

    file_path = os.path.join(model.model_dir, 'activations.jpg')
    imsave(file_path, activations_image)


def visualize_separation(model, all_samples, all_labels):
    last_activations = model.activations(layer_index=-2, samples=all_samples)[0]
    last_activations = scale_to_unit_interval(last_activations)
    x = last_activations[:, 0]
    y = last_activations[:, 1]

    plt.scatter(x, y, c=all_labels[:, 1])
    plt.plot([0, 1], [0, 1], c="green")

    file_path = os.path.join(model.model_dir, 'separation.jpg')
    plt.savefig(file_path)
    plt.clf()


def visualize_roc(model, all_samples, all_labels):
    from sklearn.metrics import roc_curve, auc

    preds = model.predict(all_samples)
    fpr, tpr, _ = roc_curve(all_labels[:, 1], preds[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], ls="--", label="ROC, AUC = {0:.2f}".format(auc_score))
    plt.legend()

    file_path = os.path.join(model.model_dir, 'roc.jpg')
    plt.savefig(file_path)
    plt.clf()


def create_tiles(X, img_shape, grid_shape, tile_spacing=(0, 0)):
    """
    :param X: (n_images, height, width)
    """

    assert len(img_shape) == 2
    assert len(grid_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [0, 0]
    out_shape[0] = (img_shape[0] + tile_spacing[0]) * grid_shape[0] - tile_spacing[0]
    out_shape[1] = (img_shape[1] + tile_spacing[1]) * grid_shape[1] - tile_spacing[1]

    # if we are dealing with only one channel
    H, W = img_shape
    Hs, Ws = tile_spacing

    # generate a matrix to store the output
    out_array = np.zeros(out_shape, dtype='float32')

    for tile_row in range(grid_shape[0]):
        for tile_col in range(grid_shape[1]):
            if tile_row * grid_shape[1] + tile_col < X.shape[0]:
                # pick the next channel
                this_x = X[tile_row * grid_shape[1] + tile_col]

                # scale the image to unit interval
                this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] = this_img
    return out_array


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar
