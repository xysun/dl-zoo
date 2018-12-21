'''
Reproduce mnist experiment in original batch normalization paper

Network:
- 3 fully-connected hidden layers, each with 100 activation units; final layer 10 units with cross-entropy loss
- BN on every hidden layer

References
- https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
- Tensorflow keras doc https://www.tensorflow.org/guide/keras
'''
import tensorflow as tf
from tensorflow.contrib.keras import layers

import numpy as np

model = tf.keras.Sequential()
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
# model.add(layers.Dense(100, activation="relu"))

model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

mnist = tf.contrib.learn.datasets.load_dataset('mnist')
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

model.fit(train_data, train_labels, epochs=50, batch_size=32) #, validation_data=(eval_data, eval_labels))
# print(model.evaluate(eval_data, eval_labels, batch_size=60))


# if __name__ == '__main__':
#     tf.app.run()
