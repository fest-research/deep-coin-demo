import os
import numpy as np
import cv2
from keras.models import load_model

from fujitsu.data_management.data_loader import load_dataset
from fujitsu.models.standard_classifier import ConvnetClassifier
from fujitsu.utils.log import setup_logger
from fujitsu.utils.visualize import create_tiles
from scipy.misc import imresize

log = setup_logger("demo")


n_classes = 2
img_width = img_height = 200
n_channels = 3
grid_shape = (6, 6)

if __name__ == "__main__":

    # load a pretrained model
    model = load_model('data/models/standard_convnet/weights.98-0.20-0.93.hdf5')
    model.save_weights('model_weights.h5')
    classifier = ConvnetClassifier(name="standard_convnet", n_classes=n_classes,
                                   n_channels=n_channels,
                                   img_width=img_width, img_height=img_height,
                                   dropout=0.5, learning_rate=0.001)
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
                if is_coin:
                    # cv2.imwrite('data/image{}.jpg'.format(i), frame)
                    cv2.circle(frame, center=(30, 30), radius=15, color=3)
                # else:
                    # # not happy, model should say coin
                    # cv2.imwrite('data/image{}.jpg'.format(i), frame)

                cv2.imshow('video', frame)

                # predict for this frame
                input_frame = np.expand_dims(frame.transpose((2, 0, 1)), axis=0)
                input_frame = np.asarray(input_frame, dtype="float32")

                # preprocessing
                input_frame /= 255
                input_frame -= data_mean

                score = model.predict(input_frame)[0]
                log.debug(score[1])
                is_coin = score[1] > 0.5

                # visualize a filter
                for i in [3, 6]:
                    activations = classifier.activations(i, input_frame)[0][0]
                    img_shape = activations.shape[1:]
                    activations_image = create_tiles(activations, img_shape=img_shape,
                                                     grid_shape=grid_shape,
                                                     tile_spacing=(10, 10))
                    activations_image = imresize(activations_image, size=(800, 800))
                    cv2.imshow('filters{}'.format(i), activations_image)

                i += 1
                counter = 0

        if cv2.waitKey(10) == 27:
            break
