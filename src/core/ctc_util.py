from keras import Input, Model
from keras.layers import Lambda


def ctc_model(inputs, output, **kwargs):
    """
    Create compilable model for input and output tensors
    :param inputs:
    :param output:
    :param kwargs:
    :return:
    """
    # Define placeholders
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    inputs_length = Input(name='inputs_length', shape=(None,), dtype='int32')

    # Define a decoder
    dec = Lambda(decode, output_shape=decode_output_shape, arguments={'is_greedy': True}, name='decoder')
    y_pred = dec([output, inputs_length])

    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")
    # Define loss as a layer
    loss = ctc([output, labels, inputs_length])

    return Model(input=[inputs, labels, inputs_length], output=[loss, y_pred])


def decode(inputs, **kwargs):
    """ Decodes a sequence of probabilities choosing the path with highest
    probability of occur

    # Arguments
        is_greedy: if True (default) the greedy decoder will be used;
        otherwise beam search decoder will be used

        if is_greedy is False:
            see the documentation of tf.nn.ctc_beam_search_decoder for moreoptions

    # Inputs
        A tuple (y_pred, seq_len) where:
            y_pred is a tensor (N, T, C) where N is the bath size, T is the
            maximum timestep and C is the number of classes (including the
            blank label)
            seq_len is a tensor (N,) that indicates the real number of
            timesteps of each sequence

    # Outputs
        A sparse tensor with the top path decoded sequence
    """

    # Little hack for load_model
    import tensorflow as tf
    is_greedy = kwargs.get('is_greedy', True)
    y_pred, seq_len = inputs

    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

    if is_greedy:
        decoded = tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    else:
        beam_width = kwargs.get('beam_width', 100)
        top_paths = kwargs.get('top_paths', 1)
        merge_repeated = kwargs.get('merge_repeated', True)

        decoded = tf.nn.ctc_beam_search_decoder(y_pred, seq_len, beam_width, top_paths, merge_repeated)[0][0]
    return decoded


def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return (y_pred_shape[:1], None)


def ctc_lambda_func(args):
    """ CTC cost function
    """
    y_pred, labels, inputs_length = args

    # Little hack for load_model
    import tensorflow as tf

    return tf.nn.ctc_loss(labels,
                          tf.transpose(y_pred, perm=[1, 0, 2]),
                          inputs_length[:, 0])
