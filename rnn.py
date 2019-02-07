'''
https://www.tensorflow.org/tutorials/sequences/text_generation but with Estimator API
'''

import functools

import numpy as np
import tensorflow as tf

from scipy.special import softmax

print(tf.__version__)

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

    return dataset.repeat()


def gru_model_fn(features, labels, mode, params):
    '''
    we could have used keras model -> model.load_weights -> tf.model_to_estimator
    but that way we lose auto Tensorboard
    although going this way we lose variable batch_size via model.build(input_shape)
    you'll see later in `predict` function I have to do an ugly hack to match the batch_size
    '''
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


    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'logits': logits})


def predict(estimator, vocab, num_generate, starting_word):
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = np.array(vocab)

    input_eval = [char2idx[c] for c in starting_word]
    # ugly hack: repeat BATCH_SIZE times,
    # because we cannot change batch_size now it's part of saved model parameters (because of embedding layer)
    input_eval_tiled = np.tile(input_eval, (BATCH_SIZE, 1))

    generated = []

    for i in range(num_generate):
        prediction = next(estimator.predict(input_fn=lambda: tf.convert_to_tensor(input_eval_tiled)))  # we take first one
        logits = prediction['logits'][-1]  # last (i.e. newly generated) character, shape(1,64)
        # sample from multinomial distribution, we use numpy because tf.multinomial returns Tensor type
        # first normalize probability
        logits_exp = np.exp(logits - np.max(logits)).astype('float64') # better stability, also see https://github.com/numpy/numpy/issues/8317
        pvals = np.true_divide(logits_exp, np.sum(logits_exp))
        print(sum(pvals[:-1]))
        predicted_ids = np.random.multinomial(n=100, pvals=pvals)
        predicted_id = np.argmax(predicted_ids)
        print("Char %d, max dice %d" % (i, max(predicted_ids)))
        generated.append(idx2char[predicted_id])

        # update input_eval, I find accumulating generate slightly better results
        input_eval.append(predicted_id)
        # input_eval = [predicted_id]
        input_eval_tiled = np.tile(input_eval, (BATCH_SIZE, 1))

    return starting_word + ''.join(generated)


def main(args):
    mode = args[0]
    assert mode in ['train', 'generate']

    text = open(DATA_PATH, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    estimator = tf.estimator.Estimator(
        model_fn=gru_model_fn,
        model_dir='tf_processing/rnn',
        config=tf.estimator.RunConfig(
            log_step_count_steps=10,
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

    if mode == 'train':
        print("Total steps to train %d" % (steps_per_epoch * EPOCHS))

        estimator.train(input_fn=lambda: train_input_fn(text, vocab, SEQ_LENGTH, BATCH_SIZE),
                        steps=steps_per_epoch * EPOCHS)

    elif mode == 'generate':
        generated = predict(estimator, vocab, num_generate=200, starting_word='ROMEO')
        print(generated)


if __name__ == '__main__':
    tf.app.run(main=main, argv=['generate'])  # use `train` or `generate`
