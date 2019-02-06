'''
https://www.tensorflow.org/tutorials/sequences/text_generation but with Estimator API
todo:
- plot loss in tensorboard
- seems not restoring weights?
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
EPOCHS = 3


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

    return dataset


def loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    tf.summary.scalar('loss', tf.reduce_mean(loss))
    return loss


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
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

    return model


def main(unused_args):
    text = open(DATA_PATH, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    model = build_model(len(vocab), EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=loss)
    print(model.summary())
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir='tf_processing/rnn',
        config=tf.estimator.RunConfig(log_step_count_steps=1)
    )

    examples_per_batch = len(text) // SEQ_LENGTH
    steps_per_epoch = examples_per_batch // BATCH_SIZE

    estimator.train(input_fn=lambda: train_input_fn(text, vocab, SEQ_LENGTH, BATCH_SIZE),
                    steps=steps_per_epoch)


if __name__ == '__main__':
    tf.app.run()
