from Model import SegNet
from dataset import parse,prepare_batch
import tensorflow as tf
from loss import loss
import numpy as np
import sys
import os
batchsize=4
imgdir="DS"
groundtruth="GT"
total_steps=10000
ckpt_dir="ckpt/"
ckpt_steps=200
load=-1
gpu=0.5
lr=1e-04

print("--loadfrom;",sys.argv[1]," --ckptdir;",sys.argv[2]," --gpu",sys.argv[3]," --lr", sys.argv[4],"save",sys.argv[5])



load=int(sys.argv[1])
ckpt_dir=sys.argv[2]
gpu=float(sys.argv[3])
lr=float(sys.argv[4])
ckpt_steps=int(sys.argv[5])


assert (os.path.exists(ckpt_dir))
assert (os.path.exists(imgdir))
assert (os.path.exists(groundtruth))



#tensor_in=tf.constant(1.0,shape=[batchsize,224,224,1],dtype=tf.float32)
segnet=SegNet(batchsize)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

train_batch=tf.placeholder(dtype=tf.float32, shape=[batchsize, 224, 224, 3])
labels=tf.placeholder(dtype=tf.int32,shape=[batchsize,224,224])
print(train_batch.get_shape().as_list())


#labels=tf.one_hot(labels,2)

logits = segnet.inference(train_batch)

print("logits=",logits.get_shape().as_list())
print("labels=",labels.get_shape().as_list())

loss_op = loss(logits, labels)


optimizer = tf.train.AdamOptimizer(lr)
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

        #print(_batch.shape)

        #_temp=tf.one_hot(indices=tf.cast(_labels, tf.int32), depth=2)
        _,loss=sess.run([train_step,loss_op], feed_dict={train_batch:_batch, labels:_labels})


        print("step", i, "Loss=",loss)

        if i % ckpt_steps==0 and i!=start:
            print("saving checkpoint ",str(i) ,".ckpt.....")

            save_path = saver.save(sess, os.path.join(ckpt_dir,str(i)+".ckpt"))







