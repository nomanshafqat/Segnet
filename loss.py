import tensorflow as tf


def loss(logits, labels):
    with tf.variable_scope("Loss"):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar("Loss", loss)
    return loss

