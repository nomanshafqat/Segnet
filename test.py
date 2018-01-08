import tensorflow as tf
from Model import SegNet
from dataset import parse,prepare_batch
from loss import loss
import numpy as np
import cv2
import sys
import os
def accuracy(sess,logits,_batch ,labels):

    softmax = tf.nn.softmax(logits,feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})
    argmax = tf.argmax(softmax, 3)

    print(sess.run(argmax))
    shape = logits.get_shape().as_list()
    n = shape[3]
    pred=sess.run(argmax)
    pred=np.array(pred)



    results=tf.subtract(labels,pred)

    incoorect=sess.run(tf.count_nonzero(results))
    total=shape[0]*shape[1]*shape[2]

    print(np.count_nonzero(pred))
    print(incoorect)


    print("error %",100*incoorect/total)
    print(pred[0]*255)

    out=cv2.multiply(pred[0],255)
    out=np.array(out).astype("uint8")
    cv2.imshow("result",out)
    cv2.imshow("original",np.array(sess.run(labels[0])).astype("uint8")*255)


    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    #equal_pixels = tf.reduce_sum())
    #total_pixels = reduce(lambda x, y: x * y, shape[:3])
    #return equal_pixels / total_pixels
    return 5

def test(load,ckpt_dir):
    batchsize = 4
    imgdir = "DS"
    groundtruth = "GT"
    gpu=0.5
    segnet = SegNet(batchsize)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch = tf.placeholder(dtype=tf.float32, shape=[batchsize, 224, 224, 3])
    labels = tf.placeholder(dtype=tf.int32, shape=[batchsize, 224, 224])
    print(train_batch.get_shape().as_list())

    # labels=tf.one_hot(labels,2)

    logits = segnet.inference(train_batch)

    softmax = tf.nn.softmax(logits)

    print("logits=", logits.get_shape().as_list())
    print("labels=", labels.get_shape().as_list())


    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(init)


        start = 0
        if load > 0:
            print("Restoring", load, ".ckpt.....")
            saver.restore(sess, os.path.join(ckpt_dir,str(load)))
            start = load

        _batch, _labels = prepare_batch(imgdir, groundtruth, batchsize)

        print(_batch.shape)

        # _temp=tf.one_hot(indices=tf.cast(_labels, tf.int32), depth=2)
        #sess.run(logits, feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})
        softmax=sess.run([softmax],feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})

        #argmax = tf.argmax(softmax, 3)

        print(softmax)
        print("softmax=", np.array(softmax).shape)

        #accuracy(sess,logits,_batch,labels)

        #accuracy()




load=int(sys.argv[1])
ckpt_dir=sys.argv[2]

print("loading", load)
print("loadingdir", ckpt_dir)


test(load,ckpt_dir)

