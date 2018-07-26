from keras import callbacks, backend as K

from core.dataset_generator import BatchGenerator
from util.rnn_util import decode


class MetaCheckpoint(callbacks.ModelCheckpoint):
    # from: https://github.com/igormq/asr-study

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

    def __init__(self, test_func, dev_batches: BatchGenerator, model, target_dir):
        super().__init__()
        self.test_func = test_func
        self.dev_batches = dev_batches
        self.model = model
        self.target_dir = target_dir

    def validate_epoch_end(self, verbose=0):
        for inputs, outputs in self.dev_batches:
            X = inputs['the_input']
            X_lengths = inputs['input_length']
            truths = inputs['source_str']

            y_pred = self.test_func([X])[0]
            sequences, probs = K.ctc_decode(y_pred, X_lengths, greedy=False)
            predictions = [decode(K.get_value(seq).reshape(-1)) for seq in sequences]

            for truth, pred in zip(truths, predictions):
                print(f'truth: {truth}, prediction: {pred}')
            break

    def on_epoch_end(self, epoch, logs=None):
        print(f'Validating epoch {epoch}')
        K.set_learning_phase(0)
        self.validate_epoch_end(verbose=1)
        K.set_learning_phase(1)
