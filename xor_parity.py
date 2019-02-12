'''
xor warmup from https://blog.openai.com/requests-for-research-2/
we are using "even parity", https://en.wikipedia.org/wiki/Parity_bit, i.e. 0 if count of 1-bits is even, 1 if odd
writeup:
- early stopping
'''

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

print(tf.__version__)

BATCH_SIZE = 8
RNN_UNITS = 24
WARMUP = 10000
BITS_MAX_LENGTH = 50
DATASET_SIZE = 100000


def generate_train_data(count, max_length, warmup):
    '''
    Generate a list containing `count` of random bit strings

    if `randomize_length` is True, each string has length uniformly drawn from [1, `length`]
    else each string is of length `length`

    warmup: generate `warmup` number of 2 bits first

    :return: list of 0,1; sorted by length
    '''
    print("Generating %d bits list of size %d, first %d are warmups" % (count, max_length, warmup))

    # generate count-warmup bits array, they have minimal size 3, sort by length, shorter first
    bits_list = [np.random.choice([0, 1], size=np.random.randint(3, max_length + 1)) for _ in range(count - warmup)]
    bits_list.sort(key=lambda e: len(e))

    # prepad 0s
    for i, bits in enumerate(bits_list):
        if len(bits) < max_length:
            bits_list[i] = np.pad(bits, pad_width=[max_length - len(bits), 0], mode='constant', constant_values=0)

    for i in range(count):
        if i < warmup:
            bits = np.random.choice([0, 1], size=2)
            # prepad 0s
            bits = np.pad(bits, pad_width=[max_length - 2, 0], mode='constant', constant_values=0)
        else:
            bits = bits_list[i - warmup]

        parity = np.count_nonzero(bits) % 2
        yield (
            np.expand_dims(bits, axis=2).astype('float32'),
            [parity]
        )


def generate_eval_data():
    while True:
        bits = np.random.choice([0, 1], size=BITS_MAX_LENGTH)
        parity = np.count_nonzero(bits) % 2
        yield (
            np.expand_dims(bits, axis=2).astype('float32'),
            [parity]
        )


def dataset_from_gen(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: generator,
        output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, 1]), tf.TensorShape([None]))
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def build_lstm_model(params):
    rnn_units = params['rnn_units']

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                rnn_units,
                return_sequences=False,
                recurrent_initializer='glorot_uniform',
                recurrent_activation='sigmoid',
                stateful=True,

            ),
            tf.keras.layers.Dense(1)
        ])

    return model


def model_fn(features, labels, mode, params):
    model = build_lstm_model(params)

    logits = model(features)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = tf.squeeze(logits)
        predictions = tf.round(tf.sigmoid(logits))
        metrics = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={"accuracy": metrics})


def main(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='tf_processing/xor',
        config=tf.estimator.RunConfig(
            log_step_count_steps=100,
            save_checkpoints_steps=100
        ),
        params={
            'batch_size': BATCH_SIZE,
            'rnn_units': RNN_UNITS
        }
    )

    steps = DATASET_SIZE // BATCH_SIZE
    
    print("Total steps to train %d" % steps)

    train_generator = generate_train_data(DATASET_SIZE, BITS_MAX_LENGTH, WARMUP)
    eval_generator = generate_eval_data()

    early_stopping_hook = tf.contrib.estimator.stop_if_higher_hook(
        estimator=estimator,
        metric_name='accuracy',
        threshold=0.98,
        run_every_steps=100,
        run_every_secs=None
    )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset_from_gen(train_generator, BATCH_SIZE), max_steps=steps,
                                        hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset_from_gen(eval_generator, BATCH_SIZE), steps=100,
                                      throttle_secs=1)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print("Final evaluation")
    estimator.evaluate(input_fn=lambda: dataset_from_gen(eval_generator, BATCH_SIZE), steps=100)


if __name__ == '__main__':
    tf.app.run(main=main)
