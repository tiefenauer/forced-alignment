"""
Some callbacks that can be used during training
"""
from os import remove
from os.path import join, exists

from keras import callbacks, backend as K

from util.rnn_util import decode


class MetaCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaCheckpoint, self).__init__(filepath, monitor='val_loss',
                                             verbose=0, save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto', period=1)

        self.filepath = filepath
        self.meta = meta or {'epochs': []}

        if training_args:
            training_args = vars(training_args)

            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        super(MetaCheckpoint, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs={}):
        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.epochs_since_last_save == 0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs',
                                          data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))


class CustomProgbarLogger(callbacks.ProgbarLogger):

    def __init__(self, count_mode='steps', stateful_metrics=None):
        super(CustomProgbarLogger, self).__init__(count_mode, ['loss', 'decoder_ler', 'val_loss', 'val_decoder_ler'])
        self.show_metrics = stateful_metrics

    def on_train_begin(self, logs=None):
        super(CustomProgbarLogger, self).on_train_begin(logs)
        if self.show_metrics:
            self.params['metrics'] = self.show_metrics


class ReportCallback(callbacks.Callback):

    def __init__(self, model, val_it, target_dir):
        super().__init__()
        self.model = model
        self.val_it = val_it
        self.inputs, self.outputs = next(self.val_it)
        self.log_file_path = join(target_dir, 'validation.log')
        if exists(self.log_file_path):
            remove(self.log_file_path)

        # create Function-graph to get decoded sequences for validation data
        # K.function([self.model.get_layer('inputs').input], [self.model.get_layer('decoder').input[0]])([X])
        input_data = model.get_layer('inputs').input
        decoder = model.get_layer('decoder').input[0]
        self.prediction_fun = K.function([input_data, K.learning_phase()], [decoder])

    def validate_epoch_end(self, epoch):
        X, Y, X_lengths, Y_lengths = self.inputs

        truths = [decode(lbl).strip() for m in list(Y.todense()) for lbl in m.tolist()]

        y_pred = self.prediction_fun([X])[0]
        sequences, probs = K.ctc_decode(y_pred, X_lengths, greedy=False)
        sequence_values = [K.get_value(seq) for seq in sequences][0]
        predictions = [decode(seq_val) for seq_val in sequence_values]

        with open(self.log_file_path, 'a') as f:
            for truth, pred in zip(truths, predictions):
                log_str = f'epoch {str(epoch).ljust(2)}: truth: {truth}, prediction: {pred}'
                print(log_str)
                f.write(log_str + '\n')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Validating epoch {epoch}')
        K.set_learning_phase(0)
        self.validate_epoch_end(epoch=epoch)
        K.set_learning_phase(1)
