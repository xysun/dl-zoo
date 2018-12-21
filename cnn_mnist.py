'''
Todo:
- implement CNN from scratch with numpy, with dropout
- visualise intermediate filters
- read original dropout paper http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
- learn how to use TensorBoard
- learn how to use TensorFlow Debugger

Q:
- how does each filter learn something differently?
    - initialised differently: can i try to initialise them with equal weight?
    - cost function is reduced by different features, so if 2 filters learn same feature it's not efficient in reducing loss
    - actually no guarantee; so use many filters
'''

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)  # default is WARN


def cnn_model_fn(features, labels, mode):
    assert mode == tf.estimator.ModeKeys.TRAIN or \
           mode == tf.estimator.ModeKeys.EVAL or \
           mode == tf.estimator.ModeKeys.PREDICT

    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])  # [batch_size, height, width, channels]
    # Q: why -1? what's original shape of features['x']?
    # A: first dimension is "batch size" for gradient descent, -1 means it's same as number of input values in features['x']
    #    basically how many examples we feed

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,  # Q: how does each filter learn to look for something different? see comment block at top
        # also this 32 goes into 'channels' dimension in output
        kernel_size=[5, 5],
        padding='same',  # for borders; pad 0s, so output is still 28x28, otherwise output will be 24x24
        activation=tf.nn.relu  # leaky_relu performs similar
    )  # output: [batch_size, 28, 28, 32]

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )  # [batch_size, 14, 14, 32]

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )  # [batch_size, 14, 14, 64]; convnet always convolute across full depth of image

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )  # [batch_size, 7, 7, 64]

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # ^ this is why "each output of a dense layer is connected to every input",
    # because it's always 1xn matmul nxm = 1xm

    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024, # output neurons
        activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(
        inputs=dropout,
        units=10  # no activation
    )

    predictions = {
        'classes': tf.argmax(logits, axis=1), # 1 because shape is [batch_size, 10]
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor') # applies softmax to logits (normalise to 1 basically)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # without sparse: feed in one hot labels

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
        # 0.001: {'accuracy': 0.7501, 'loss': 1.8770982, 'global_step': 1140}
        # 0.005: {'accuracy': 0.9181, 'loss': 0.2829149, 'global_step': 1000}
        # 0.01: {'accuracy': 0.9488, 'loss': 0.1742343, 'global_step': 1000}
        # 0.02: {'accuracy': 0.9631, 'loss': 0.1179952, 'global_step': 1000}
        # 0.05: {'accuracy': 0.9803, 'loss': 0.06186451, 'global_step': 1000}
        # 0.1: {'accuracy': 0.9825, 'loss': 0.05313045, 'global_step': 1000}
        # 0.2: {'accuracy': 0.9851, 'loss': 0.042221442, 'global_step': 1000}
        # 0.5: {'accuracy': 0.9797, 'loss': 0.05582719, 'global_step': 1000}
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
            # keeps track of steps seen by graph, you can reset by clear the checkpoint dir
            # default behaviour is incremental
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)  # overload meh

    eval_matric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_matric_ops)  # multi-branch return meh


def main(unused_args): # has to take one arg
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images # shape is (50000,784)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, # returns EstimatorSpec
        model_dir='/tmp/mnist_cnn' # clear or change to train from scratch
    )

    tensors_to_log = {'probabilities':'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn( # feed numpy input into model
        x = {'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None, # run forever until step
        # Q: epoch vs steps:
        # A: epoch is number of passes through full data, steps is number of gradient descents, one descent per batch
        #    so if epoch = 1, we need at least 50k/100 = 500 steps to do a full pass
        #    but if you feed in steps = 200, it will stop after 200
        #    setting epoch to None, it will stop after `steps`
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run() # requires a main(args)
