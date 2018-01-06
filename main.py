from Model import SegNet
from dataset import parse,prepare_batch
import tensorflow as tf
from loss import loss

batchsize=4
imgdir="/Users/nomanshafqat/Desktop/DIP/IMAGES/DRIVE"
groundtruth="GT"



tensor_in=tf.constant(1.0,shape=[batchsize,224,224,1],dtype=tf.float32)
segnet=SegNet(batchsize)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

batch,labels=prepare_batch(imgdir,groundtruth,batchsize)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)

logits = segnet.inference(batch)

optimizer = tf.train.AdamOptimizer(1e-04)
loss_op = loss(logits, onehot_labels)
train_step = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

with tf.Session(config=session_config) as sess:
    sess.run(init)

    for i in range(1,100):
        #print(sess.run(batch))
        sess.run(train_step,feed_dict={})
        print("step", i, "Loss=",sess.run(loss_op))



