def evaluate_model(model, test_it):
    print(f'Evaluating on {len(test_it)} batches ({test_it.n} speech segments)')
    model.evaluate_generator(test_it)

    # for inputs, outputs in test_batches.next_batch():
    #     X_lengths = inputs['input_length']
    #     Y_pred = model.predict(inputs)
    #     res = tf.keras.backend.ctc_decode(Y_pred, X_lengths)
    #     print(f'prediction: {decode(y_pred)}')
    #     print(f'actual: {decode(y)}')
