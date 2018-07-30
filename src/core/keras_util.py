import keras
import tensorflow as tf
from keras import backend as K
from keras.utils import get_custom_objects

from core.ctc_util import clipped_relu


def load_model(model_path):
    update_custom_objects()
    model = keras.models.load_model(model_path)

    return model


def update_custom_objects():
    get_custom_objects().update({'clipped_relu': clipped_relu})
    get_custom_objects().update({'ctc_dummy_loss': ctc_dummy_loss})
    get_custom_objects().update({'decoder_dummy_loss': decoder_dummy_loss})
    get_custom_objects().update({'ler': ler})


def ctc_dummy_loss(y_true, y_pred):
    """
    CTC-Loss for Keras. Because Keras has no CTC-loss built-in, we create this dummy.
    We simply use the output of the RNN, which corresponds to the CTC loss.
    """
    return y_pred


def decoder_dummy_loss(y_true, y_pred):
    """
    Loss of the decoded sequence. Since this is not optimized, we simply return a zero-Tensor
    """
    return K.zeros((1,))


def ler(y_true, y_pred, **kwargs):
    """
    LER-Loss (see https://www.tensorflow.org/api_docs/python/tf/edit_distance)
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, normalize=True, **kwargs))