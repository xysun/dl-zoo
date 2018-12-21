'''
Reproduce mnist experiment in original batch normalization paper

Network:
- 3 fully-connected hidden layers, each with 100 activation units; final layer 10 units with cross-entropy loss
- BN on every hidden layer

References
- https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
- Tensorflow keras doc https://www.tensorflow.org/guide/keras
'''
import numpy as np
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)


def nn_model_fn(features, labels, mode):
    assert mode == tf.estimator.ModeKeys.TRAIN or \
           mode == tf.estimator.ModeKeys.EVAL or \
           mode == tf.estimator.ModeKeys.PREDICT

    input_layer = tf.reshape(features['x'], [-1, 28 * 28])

    dense1 = tf.layers.dense(
        inputs=input_layer,
        units=100,
        activation=tf.nn.relu
    )

    dense2 = tf.layers.dense(
        inputs=dense1,
        units=100,
        activation=tf.nn.relu
    )

    logits = tf.layers.dense(
        inputs=dense2,
        units=10
    )

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_matric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'], name="accuracy"
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_matric_ops)


def main(unused_args):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=nn_model_fn,
        model_dir='/tmp/mnist_cnn'
    )

    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=60,
        num_epochs=None,
        shuffle=True
    )

    for i in range(10):

        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1,
            # hooks=[logging_hook]
        )

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

        with open('no_bn.csv', 'a') as f:
            f.write("%d,%s\n" % (i * 50, eval_results['accuracy']))

        print(i, eval_results)


if __name__ == '__main__':
    tf.app.run()
