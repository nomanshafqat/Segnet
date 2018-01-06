import tensorflow as tf


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy, name='loss')


