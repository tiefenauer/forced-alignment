import keras
import tensorflow as tf
from keras import backend as K, Model
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from util.brnn_util import decoder_lambda_func, clipped_relu, ctc_dummy_loss, decoder_dummy_loss, ler


def load_model(model_path):
    update_custom_objects()
    return keras.models.load_model(model_path)


def update_custom_objects():
    get_custom_objects().update({'clipped_relu': clipped_relu})
    get_custom_objects().update({'ctc_dummy_loss': ctc_dummy_loss})
    get_custom_objects().update({'decoder_dummy_loss': decoder_dummy_loss})
    get_custom_objects().update({'ler': ler})


def load_model_for_prediction(model_path):
    """
    Create a new model from the pre-trained tensors of a saved model. The new model will only use the input features (X)
    and the sequence lengths (X_length) as input and only produce the decoded sequence as output. The output layer
    with the predictions will be a dense Tensor that was converted from the SparseTensor conttaining the non-greedily
    decoded sequences with beam search.

    This procedure is needed because the pre-trained, saved BRNN uses SparseTensors but those can somehow not be used
    when making predictions with `model.predict(...)`

    :param model_path: path to file with saved model
    :return: model that can be used to make predictions.
    """
    model = load_model(model_path)

    # extract tensors to use as input
    inputs = model.get_layer('inputs').input
    inputs_length = model.get_layer('inputs_length').input

    # extract tensor to use as output and convert to dense
    y_pred = model.get_layer('decoder').input[0]
    dec = Lambda(decoder_lambda_func, name='beam_search')
    to_dense_layer = Lambda(to_dense, name="to_dense")
    y_pred = dec([y_pred, inputs_length])
    y_pred = to_dense_layer(y_pred)

    model = Model(inputs=[inputs, inputs_length], outputs=[y_pred])

    return model


def load_model_for_evaluation(model_path):
    """
    Create a new model from the pre-trained tensors of a saved model. The new model will only use the input features (X)
    and the sequence lengths (X_length) as input and only produce the decoded sequence as output. The output layer
    with the predictions will be a dense Tensor that was converted from the SparseTensor conttaining the non-greedily
    decoded sequences with beam search.
    :param model_path:
    :return:
    """
    model = load_model(model_path)

    dec_layer = model.get_layer('decoder')
    dec = Lambda(decoder_lambda_func, name='beam_search')
    y_pred = dec(dec_layer.input)

    model = Model(inputs=model.inputs, outputs=[model.outputs[0], y_pred])

    # Freezing layers
    for l in model.layers:
        l.trainable = False

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(optimizer=opt,
                  loss={'ctc': ctc_dummy_loss, 'beam_search': decoder_dummy_loss},
                  metrics={'beam_search': ler},
                  loss_weights=[1, 0])

    return model


def to_dense(x):
    if K.is_sparse(x):
        return tf.sparse_tensor_to_dense(x, default_value=-1)
    return x
