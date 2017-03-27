"""
This module is a standard convnet classifier, written in tensorflow.
"""
import os

from keras import callbacks
import keras.backend as K

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator


class ConvnetClassifier(object):
    def __init__(self, name, network_fn, n_classes, n_channels, img_width, img_height,
                 dropout=0.5, learning_rate=0.001):
        """__init__

        :param name: model name, will be used as a model ID to create the model directory
        :param network_fn: function of the neural network structure:
            Usage: output_layer = network_fn(input_layer, dropout)
        :param n_classes: how many classes do we want to distinguish from the data
        :param n_channels: channels in the images, the model uses the "channels_first" convention
            batch_inputs_shape = (batch_size, n_channels, height, width)
        :param img_width: width of the processed images
        :param img_height: height of the processed images
        :param dropout: dropout parameter, passed to the network_fn
        :param learning_rate: desired learning rate for the gradient decent optimization
        """
        self.name = name
        self.model_dir = os.path.join(os.path.dirname(__file__),
                                      "../../data/models/{}".format(self.name))
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_width = img_width
        self.img_height = img_height
        self.dropout = dropout

        # define the symbolic variables
        inputs = Input(shape=(self.n_channels, self.img_height, self.img_width))
        network = inputs

        # apply the network
        network = network_fn(inputs, self.dropout)

        # form the classification layers
        network = Dense(self.n_classes)(network)
        predictions = Activation(activation=K.softmax)(network)

        # Define the model
        self.model = Model(input=inputs, output=predictions)

        # define the gradient descent optimization
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model._make_test_function()
        self.model._make_train_function()
        self.model._make_predict_function()

        # callbacks (add-ons during training)
        # visualizations
        self.tensor_board = callbacks.TensorBoard(log_dir=os.path.join(self.model_dir, 'logs'),
                                                  histogram_freq=5,
                                                  write_graph=False,
                                                  write_images=False)

        # weight checkpointing
        checkpoint_path = os.path.join(self.model_dir,
                                       "weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5")
        self.checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_loss', save_best_only=True)

        # some utility functions for inspection
        self._activation_functions = []
        for layer in self.model.layers:
            self._activation_functions.append(K.function(inputs=[K.learning_phase()] + self.model.inputs,
                                                         outputs=[layer.output]))

    def get_name(self):
        return self.name

    def train(self, train_samples, train_labels,
              n_epochs=100, batch_size=8, validation_split=0.3, class_weight=None):
        """train

        :param train_samples: shape (N, n_channels, img_height, img_width)
        :param train_labels: shape (N, )
        :param n_epochs: for how many epochs to train
        :param batch_size: desired size of the mini-batches
        :param validation_split: percentage of the train data to use for validation
        :param class_weight: (optional) dictionary of weights for the different classes,
            (e.g. if the classification of one is more important)
        """
        # ensure the model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # split a validation set
        val_size = int(validation_split * train_samples.shape[0])
        val_samples = train_samples[:val_size]
        val_labels = train_labels[:val_size]
        train_samples = train_samples[val_size:]
        train_labels = train_labels[val_size:]

        # dynamic data augmentation & minibatch feeding

        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            channel_shift_range=0.2,
            rotation_range=180,
            zoom_range=0.3,
            horizontal_flip=False)

        # datagen = ImageDataGenerator(
        # width_shift_range=0.0,
        # height_shift_range=0.0,
        # channel_shift_range=0.0,
        # rotation_range=0,
        # zoom_range=0,
        # horizontal_flip=False)

        datagen.fit(train_samples)

        # start the training
        self.model.fit_generator(datagen.flow(train_samples, train_labels, batch_size=batch_size),
                                 samples_per_epoch=train_samples.shape[0],
                                 nb_epoch=n_epochs,
                                 class_weight=class_weight,
                                 validation_data=(val_samples, val_labels),
                                 callbacks=[self.checkpoint,
                                            self.tensor_board])

    def predict(self, samples, batch_size=32):
        """predict

        :param samples: images to predict for, shape (N, n_channels, img_height, img_width)
        :param batch_size: mini-batch size to use during prediction
        :returns: the predictions (classification scores)
        """
        return self.model.predict(samples, batch_size=batch_size, verbose=0)

    def activations(self, layer_index, samples):
        """
        activations - get the output of any hidden layer in the network

        :param layer_index: which layer's output ( 0-based index)
        :param samples: image samples for which the hidden outputs should be produced
        :returns: the hidden outputs
        """
        return self._activation_functions[layer_index]([0, samples])
