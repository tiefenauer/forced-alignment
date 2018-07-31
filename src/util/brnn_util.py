from keras import Input
from keras.layers import TimeDistributed, Dense, Activation, Dropout, Bidirectional, SimpleRNN
from keras.utils import get_custom_objects

from core.ctc_util import clipped_relu, ctc_model


def deep_speech_model(num_features, num_hidden=2048, dropout=0.1, num_classes=28):
    """
    Deep Speech model with architecture as described in the paper:
        5 Layers (3xFC + 1xBRNN + 1xFC) with Dropout applied to FC layer
    The output contains 28 classes: a..z, space, blank

    Differences to the original setup:
        * We are not translating the raw audio files by 5 ms (Sec 2.1 in [1])
        * We are not striding the RNN to halve the timesteps (Sec 3.3 in [1])
        * We are not using frames of context

    Reference: [1] https://arxiv.org/abs/1412.5567
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})
    x = Input(name='inputs', shape=(None, num_features))
    o = x

    # First layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Second layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Third layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fourth layer
    o = Bidirectional(SimpleRNN(num_hidden, return_sequences=True,
                                dropout=dropout,
                                activation=clipped_relu,
                                kernel_initializer='he_normal'), merge_mode='sum')(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fifth layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Output layer
    o = TimeDistributed(Dense(num_classes, name='y_pred', activation='softmax'), name='out')(o)

    return ctc_model(x, o)