"""
Deep learning of a simple coin example.
"""
import os
import threading
import time

import yaml
import numpy as np

from fujitsu.data_management.data_loader import load_dataset
from fujitsu.models.standard_classifier import ConvnetClassifier
from fujitsu.networks import get_network
from fujitsu.utils.log import setup_logger

log = setup_logger("main")

# visualization
grid_shape = (4, 4)


def inspect_hidden_weights(model, test_sample, all_samples, all_labels):
    from fujitsu.utils.visualize import visualize_separation, visualize_roc

    # wait for tensorflow to compile the model
    time.sleep(10)

    while 1:
        # visualize data before the softmax
        visualize_separation(model, all_samples, all_labels)

        # create an ROC curve
        visualize_roc(model, all_samples, all_labels)
        time.sleep(10)


if __name__ == '__main__':
    # load the model config
    with open('config.yaml', mode='rb') as f:
        config = yaml.load(f)

    # load the data
    data, mean = load_dataset(os.path.join(os.path.dirname(__file__), "data"))
    np.save('data/data_mean.npy', mean)

    log.debug("Train data shape: {}".format(data['X_train'].shape))

    # initialize the model
    classifier = ConvnetClassifier(name=config['model']['name'],
                                   network_fn=get_network(config['training']['network']),
                                   n_classes=config['model']['n_classes'],
                                   n_channels=config['data']['n_channels'],
                                   img_width=config['data']['img_width'],
                                   img_height=config['data']['img_height'],
                                   dropout=config['training']['dropout'],
                                   learning_rate=config['training']['learning_rate'])

    # continuously inspect the performance by plotting ROC curves and
    # separations
    test_sample = data['X_test'][[np.argmax(data['y_test'][:, 1])]]
    all_samples = data['X_test']
    all_labels = data['y_test']
    try:
        threading.Thread(target=inspect_hidden_weights,
                         args=(classifier, test_sample, all_samples, all_labels)).start()
    except:
        log.error("unable to start thread for hidden activation inspection")

    # start the training
    classifier.train(train_samples=data['X_train'],
                     train_labels=data['y_train'],
                     batch_size=config['training']['batch_size'],
                     n_epochs=config['training']['epochs'])

    # save the model's config for reproducability
    with open(os.path.join(classifier.model_dir, "config.yaml"), mode='w') as f:
        yaml.dump(config, f, default_flow_style=False)
