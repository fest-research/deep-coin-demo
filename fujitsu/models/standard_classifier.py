"""
This module is a standard convnet classifier, written in tensorflow.
"""
import os

from keras import callbacks
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


class ConvnetClassifier(object):
    def __init__(self, name, n_classes, n_channels, img_width, img_height,
                 dropout=0.5, learning_rate=0.001):
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
        network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        network = Conv2D(16, 3, 3, border_mode='valid',
                         activation='relu', dim_ordering='th')(network)
        network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        network = Dropout(p=self.dropout)(network)

        network = Conv2D(32, 3, 3, border_mode='valid',
                         activation='relu', dim_ordering='th')(network)
        network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        network = Dropout(p=self.dropout)(network)

        # network = Conv2D(64, 3, 3, border_mode='valid',
        # activation='relu', dim_ordering='th')(network)
        # network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        # network = Dropout(p=self.dropout)(network)

        # network = Conv2D(8, 3, 3, border_mode='valid',
        # activation='relu', dim_ordering='th')(network)
        # network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
        # network = Dropout(p=self.dropout)(network)

        # network = Conv2D(8, 3, 3, border_mode='valid',
        # activation='relu', dim_ordering='th')(network)
        # network = Dropout(p=self.dropout)(network)

        # define the classification objective
        network = Flatten()(network)
        network = Dense(16, activation='relu')(network)
        network = Dropout(p=self.dropout)(network)
        network = Dense(self.n_classes)(network)
        predictions = Activation(activation=K.softmax)(network)

        self.model = Model(input=inputs, output=predictions)

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model._make_test_function()
        self.model._make_train_function()
        self.model._make_predict_function()

        # callbacks (during training)
        self.tensor_board = callbacks.TensorBoard(log_dir='./data/training_logs',
                                                  histogram_freq=5,
                                                  write_graph=False,
                                                  write_images=False)

        checkpoint_path = os.path.join(self.model_dir,
                                       "weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5")
        self.checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_loss', save_best_only=True)

        # some utility functions for inspection
        self._activation_functions = []
        for layer in self.model.layers:
            self._activation_functions.append(K.function(inputs=[K.learning_phase()] + self.model.inputs,
                                                         outputs=[layer.output]))

    def train(self, train_samples, train_labels,
              n_epochs=100, batch_size=8, validation_split=0.3, class_weight=None):
        # ensure the model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # split a validation set
        val_size = int(validation_split * train_samples.shape[0])
        val_samples = train_samples[:val_size]
        val_labels = train_labels[:val_size]
        train_samples = train_samples[val_size:]
        train_labels = train_labels[val_size:]

        # dynamic data augmentation
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            channel_shift_range=0.2,
            zoom_range=0.3,
            horizontal_flip=False)

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
        return self.model.predict(samples, batch_size=batch_size, verbose=0)

    def activations(self, layer_index, samples):
        return self._activation_functions[layer_index]([0, samples])
