import os
import numpy as np
import cv2
import yaml
import click
from keras.models import load_model

from fujitsu.data_management.data_loader import load_dataset
from fujitsu.models.standard_classifier import ConvnetClassifier
from fujitsu.utils.log import setup_logger
from fujitsu.utils.visualize import create_tiles
from fujitsu.networks import get_network
from scipy.misc import imresize

log = setup_logger("demo")
grid_shape = (6, 6)

@click.command()
@click.option('--modeldir', prompt='Enter path to the directory of the model to demo',
              help='path to the directory of the model you want to demo with')
@click.option('--weights', prompt='Enter weights filename',
              help='Name of the weights file to load for the model.')
def run_demo(modeldir, weights):
    # load the model config
    with open(os.path.join(modeldir, 'config.yaml'), mode='rb') as f:
        config = yaml.load(f)

    # load a pretrained model
    model = load_model(os.path.join(modeldir, weights))
    model.save_weights('model_weights.h5')

    classifier = ConvnetClassifier(name=config['model']['name'],
                                   network_fn=get_network(config['training']['network']),
                                   n_classes=config['model']['n_classes'],
                                   n_channels=config['data']['n_channels'],
                                   img_width=config['data']['img_width'],
                                   img_height=config['data']['img_height'],
                                   dropout=config['training']['dropout'],
                                   learning_rate=config['training']['learning_rate'])
    classifier.model.load_weights('model_weights.h5')
    data, data_mean = load_dataset(os.path.join(os.path.dirname(__file__), "data"))

    # test on the test set
    result = model.evaluate(data['X_test'], data['y_test'])
    log.info("Model performance: {}".format(result))

    # work with the video
    cap = cv2.VideoCapture(0)
    i = 0
    counter = 0
    is_coin = False
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                counter += 1
                # crop the frame to something smaller
                half_size = 100
                center_y = frame.shape[0] // 2
                center_x = frame.shape[1] // 2
                frame = frame[center_y - half_size:center_y + half_size,
                              center_x - half_size:center_x + half_size]

                # predict for this frame
                input_frame = np.expand_dims(frame.transpose((2, 0, 1)), axis=0)
                input_frame = np.asarray(input_frame, dtype="float32")

                # preprocessing
                input_frame /= 255
                input_frame -= data_mean

                score = model.predict(input_frame)[0]
                log.debug(score[1])
                is_coin = score[1] > 0.5

                # visualize the prediction
                if is_coin:
                    cv2.putText(frame, "coin", org=(20, 20),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1, color=(125, 255, 125),
                                thickness=2)
                else:
                    cv2.putText(frame, "no coin", org=(20, 20),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1, color=(255, 125, 125),
                                thickness=2)

                cv2.imshow('video', frame)

                # visualize a filter
                # for i in [3, 6]:
                # activations = classifier.activations(i, input_frame)[0][0]
                # img_shape = activations.shape[1:]
                # activations_image = create_tiles(activations, img_shape=img_shape,
                # grid_shape=grid_shape,
                # tile_spacing=(10, 10))
                # activations_image = imresize(activations_image, size=(800, 800))
                # cv2.imshow('filters{}'.format(i), activations_image)

                i += 1
                counter = 0

        if cv2.waitKey(10) == 27:
            break

if __name__ == "__main__":
    # get the command line parameters
    run_demo()

