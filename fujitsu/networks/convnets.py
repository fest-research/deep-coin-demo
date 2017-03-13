import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation


def network0(inputs, dropout):
    network = inputs

    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)

    network = Flatten()(network)
    return network


def network1(inputs, dropout):
    network = inputs

    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)

    network = Flatten()(network)
    network = Dense(16)(network)
    network = Activation(K.relu)(network)
    network = Dropout(p=dropout)(network)

    return network


def network2(inputs, dropout):
    network = inputs

    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)

    network = Conv2D(16, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Flatten()(network)
    network = Dense(16)(network)
    network = Activation(K.relu)(network)
    network = Dropout(p=dropout)(network)

    return network


def network3(inputs, dropout):
    network = inputs

    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)

    network = Conv2D(16, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Conv2D(32, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Flatten()(network)
    network = Dense(16)(network)
    network = Activation(K.relu)(network)
    network = Dropout(p=dropout)(network)

    return network


def network4(inputs, dropout):
    network = inputs

    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)

    network = Conv2D(16, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Conv2D(32, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Conv2D(64, 3, 3, border_mode='valid', dim_ordering='th')(network)
    network = Activation(activation=K.relu)(network)
    network = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th')(network)
    network = Dropout(p=dropout)(network)

    network = Flatten()(network)
    network = Dense(16)(network)
    network = Activation(K.relu)(network)
    network = Dropout(p=dropout)(network)

    return network
