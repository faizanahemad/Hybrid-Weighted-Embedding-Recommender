from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from .logging import getLogger
import numpy as np


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, epochs, batch_size, n_examples, divisor=10):
        super(LRSchedule, self).__init__()
        self.start_lr = lr
        self.lrs = []
        self.log = getLogger(type(self).__name__)
        self.step = 0
        self.dtype = tf.float64
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.lr = lr
        self.divisor = divisor

    def __call__(self, step):
        steps_per_epoch = int(np.ceil(self.n_examples / self.batch_size))
        total_steps = steps_per_epoch * self.epochs
        lr = self.start_lr
        step = self.step
        div = self.divisor
        new_lr = np.interp(float(K.eval(step)), [0, total_steps / 3, 0.8 * total_steps, total_steps],
                           [lr / div, lr, lr / div, lr / (2*div)])
        self.lrs.append(new_lr)
        self.step += 1
        self.lr = new_lr
        return new_lr


def resnet_layer_with_content(n_dims, n_out_dims, dropout, kernel_l2, depth=2):
    assert n_dims >= n_out_dims

    def layer(x, content=None):
        if content is not None:
            h = K.concatenate([x, content])
        else:
            h = x
        for i in range(1, depth + 1):
            dims = n_dims if i < depth else n_out_dims
            h = keras.layers.Dense(dims, activation="linear", kernel_initializer=ScaledGlorotNormal(),
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(h)
            h = tf.keras.activations.relu(h, alpha=0.1)
            # h = tf.keras.layers.BatchNormalization()(h)
        if x.shape[1] != n_out_dims:
            x = keras.layers.Dense(n_out_dims, activation="linear", kernel_initializer=ScaledGlorotNormal(),
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(x)
        x = h + x
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    return layer


class ScaledGlorotNormal(tf.keras.initializers.VarianceScaling):
    def __init__(self, scale=1.0, seed=None):
        super(ScaledGlorotNormal, self).__init__(
            scale=scale,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

