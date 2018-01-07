from Model import SegNet
from dataset import parse,prepare_batch
import tensorflow as tf
from loss import loss
import numpy as np
batchsize=4
imgdir="/Users/nomanshafqat/Desktop/DIP/IMAGES/DRIVE"
groundtruth="GT"
total_steps=10000
ckpt_dir="/Users/nomanshafqat/Desktop/DIP/ckpt/"
ckpt_steps=20
load=140

#tensor_in=tf.constant(1.0,shape=[batchsize,224,224,1],dtype=tf.float32)
segnet=SegNet(batchsize)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

train_batch=tf.placeholder(dtype=tf.float32, shape=[batchsize, 224, 224, 3])
labels=tf.placeholder(dtype=tf.int32,shape=[batchsize,224,224])
print(train_batch.get_shape().as_list())

#onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=1)

logits = segnet.inference(train_batch)
loss_op = loss(logits, labels)

optimizer = tf.train.AdamOptimizer(1e-04)
train_step = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session(config=session_config) as sess:
    sess.run(init)
    start=0
    if load > 0:
        print("Restoring", load ,".ckpt.....")
        saver.restore(sess, ckpt_dir+str(load)+".ckpt")
        start=load

    for i in range(start,total_steps):
        #print(sess.run(batch))

        _batch, _labels = prepare_batch(imgdir, groundtruth, batchsize)

        print(_batch.shape)


        _,loss=sess.run([train_step,loss_op], feed_dict={train_batch:sess.run(_batch), labels:sess.run(_labels)})


        print("step", i, "Loss=",loss)

        if i % ckpt_steps==0 and i!=start:
            print("saving checkpoint ",str(i) ,".ckpt.....")

            save_path = saver.save(sess, ckpt_dir+str(i)+".ckpt")






