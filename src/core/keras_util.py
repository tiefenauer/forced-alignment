import keras
import tensorflow as tf
from keras import backend as K, Model
from keras.layers import Lambda
from keras.utils import get_custom_objects

from core.ctc_util import clipped_relu, decoder_lambda_func, decode_output_shape


def load_model(model_path):
    update_custom_objects()

    # load model from file
    model = keras.models.load_model(model_path)

    # create new model requiring only the features (X) and the sequence lengths (X_lengths) and outputting only
    # the predicted sequence (Y_pred). This simplified model can be used to make predictions.
    inputs = model.get_layer('inputs').input
    inputs_length = model.get_layer('inputs_length').input

    # convert prediction layer to dense layer because Keras does not accept SparseTensors in prediction
    y_pred = model.get_layer('decoder').input[0]
    dec = Lambda(decoder_lambda_func, output_shape=decode_output_shape, name='beam_search')
    to_dense_layer = Lambda(to_dense, output_shape=to_dense_output_shape, name="to_dense")
    y_pred = dec([y_pred, inputs_length])
    y_pred = to_dense_layer(y_pred)

    model = Model(inputs=[inputs, inputs_length], outputs=[y_pred])

    return model


def to_dense(x):
    if K.is_sparse(x):
        return tf.sparse_tensor_to_dense(x, default_value=-1)
    return x


def to_dense_output_shape(input_shape):
    return input_shape


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
