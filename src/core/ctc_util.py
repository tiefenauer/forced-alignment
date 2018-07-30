from keras import Input, Model
from keras import backend as K
from keras.activations import relu
from keras.layers import Lambda


def ctc_model(inputs, output, **kwargs):
    """
    Create compilable model for input and output tensors
    :param inputs: input tensors of the model
    :param output: output tensor of the model
    :param kwargs: other arguments
    :return:
    """
    # Input placeholders for true labels and input sequence lengths (need to be fed during training/validation)
    labels = Input(name='labels', shape=(None,), dtype='int32', sparse=True)
    inputs_length = Input(name='inputs_length', shape=[1], dtype='int32')
    labels_length = Input(name='labels_length', shape=[1], dtype='int32')

    # Output layer for decoded label (values are integers)
    dec = Lambda(decoder_lambda_func, output_shape=decode_output_shape, arguments={'is_greedy': False}, name='decoder')
    y_pred = dec([output, inputs_length])

    # Output layer for CTC-loss
    ctc = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")
    loss = ctc([labels, output, inputs_length, labels_length])

    return Model(inputs=[inputs, labels, inputs_length, labels_length], outputs=[loss, y_pred])


def decoder_lambda_func(args, is_greedy=True, beam_width=100, top_paths=1, merge_repeated=True):
    """
    CTC-Decoder function. Decodes a batch by evaluationg sequences of probabilities either using best-path
    (greedy) or beam-search.
    :param y_pred: (batch_size, T_x, num_classes)
        tensor containing the probabilities of each character for each time step of each sequence in the batch
        - batch_size: number of sequences in batch
        - T_x: number of time steps in the current batch (maximum number of time steps over all batch sequences)
        - num_classes: number of characters (i.e. number of probabilities per time step)
    :param seq_len: (batch_size,)
        tensor containing the lenghts of each sequence in the batch (they have been padded)
    :param is_greedy: Flag for best-path or beam search decoding
        If True decoding will be done following the character with the highest probability in each
        time step (best-path decoding).
        If False beam search decoding will be done. The beam width, number of top paths
        and whether to merge repeated prefixes are specified in their respective kwargs
        --> see https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
    :param beam_width: number of paths to follow simultaneously. Only used if is_greedy=False
    :param top_paths: number of paths to return. Only used if is_greedy=False
    :param merge_repeated: whether to merge/sum up partial paths that lead to the same prefix
    :return A sparse tensor with the top_paths decoded sequence
    """
    # hack: we need to import tensorflow here again to make keras.engine.saving.load_model work...
    import tensorflow as tf

    y_pred, seq_len = args
    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])  # time major

    if is_greedy:
        return tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    return tf.nn.ctc_beam_search_decoder(y_pred, seq_len, beam_width, top_paths, merge_repeated)[0][0]


def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return y_pred_shape[:1], None


def ctc_lambda_func(args):
    """
    CTC cost function. Calculates the CTC cost over a whole batch.
    Since this is used in a lambda layer and Keras calls functions with an arguments tuple (a,b,c,...)
    and not *(a,b,c,...) the function's parameters must be unpacked inside the function. The parameters are as follows
    :param y_pred: (batch_size, T_x, num_classes)
        tensor containing the probabilities of each character for each time step of each sequence in the batch
        - batch_size: number of sequences in batch
        - T_x: number of time steps in the current batch (maximum number of time steps over all batch sequences)
        - num_classes: number of characters (i.e. number of probabilities per time step)
    :param labels: (batch_size, T_x)
        tensor containing the true labels (encoded as integers)
    :param inputs_length: (batch_size,)
        tensor containing the lenghts of each sequence in the batch (they have been padded)
    :return: tensor for the CTC-loss
    """

    # hack: we need to import tensorflow here again to make keras.engine.saving.load_model work...
    import tensorflow as tf
    y_true, y_pred, input_length, label_length = args
    y_true = tf.sparse_tensor_to_dense(y_true)
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def clipped_relu(x):
    return relu(x, max_value=20)
