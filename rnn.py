'''
https://www.tensorflow.org/tutorials/sequences/text_generation but with Estimator API
todo:
- add evaluate loop
- try use ordinary MLP

'''

import functools

import numpy as np
import tensorflow as tf

'''
# run this to download data: 
data_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
'''
DATA_PATH = '/Users/xiayunsun/.keras/datasets/shakespeare.txt'
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 1


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def train_input_fn(text, vocab, seq_length, batch_size):
    char2idx = {c: i for i, c in enumerate(vocab)}

    text_as_int = np.array([char2idx[c] for c in text])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset.repeat()


def model_fn(features, labels, mode, params):
    vocab_size = params['vocab_size']
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    rnn_units = params['rnn_units']

    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            rnn(
                rnn_units,
                return_sequences=True,  # important
                recurrent_initializer='glorot_uniform',
                stateful=True  # todo: tweak
            ),
            tf.keras.layers.Dense(vocab_size)
        ])

    logits = model(features)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    tf.summary.scalar('loss', loss)

    assert(mode == tf.estimator.ModeKeys.TRAIN)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(unused_args):
    text = open(DATA_PATH, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir='tf_processing/rnn',
        config=tf.estimator.RunConfig(
            log_step_count_steps=50,
            save_checkpoints_steps=50
        ),
        params={
            'vocab_size': len(vocab),
            'embedding_dim': EMBEDDING_DIM,
            'batch_size': BATCH_SIZE,
            'rnn_units': RNN_UNITS
        }
    )

    examples_per_batch = len(text) // SEQ_LENGTH
    steps_per_epoch = examples_per_batch // BATCH_SIZE

    print("Total steps to train %d" % (steps_per_epoch * EPOCHS))

    estimator.train(input_fn=lambda: train_input_fn(text, vocab, SEQ_LENGTH, BATCH_SIZE),
                    steps=steps_per_epoch * EPOCHS)


if __name__ == '__main__':
    tf.app.run()
