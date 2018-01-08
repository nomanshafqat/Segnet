import tensorflow as tf
from Model2 import SegNet
from dataset import parse,prepare_batch
from loss import loss
import numpy as np
import cv2
import sys
import os
def accuracy(results ,labels):


    results=np.array(results)
    shape=results.shape


    subtrac=np.subtract(results,labels)


    print(results)
    print(labels)

    incoorect=np.count_nonzero(subtrac)
    total=shape[0]*shape[1]*shape[2]

    print(incoorect)


    print("error %",100*incoorect/total)
    print(results[0]*255)


    cv2.imwrite("results/result.png",np.array(results[0]).astype("uint8")*255)
    cv2.imwrite("results/original.png",np.array(labels[0]).astype("uint8")*255)

    cv2.imwrite("results/result1.png", np.array(results[1]).astype("uint8") * 255)
    cv2.imwrite("results/original1.png", np.array(labels[1]).astype("uint8") * 255)

    cv2.imwrite("results/result2.png", np.array(results[2]).astype("uint8") * 255)
    cv2.imwrite("results/original2.png", np.array(labels[2]).astype("uint8") * 255)

    cv2.imwrite("results/result3.png", np.array(results[2]).astype("uint8") * 255)
    cv2.imwrite("results/original3.png", np.array(labels[2]).astype("uint8") * 255)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    #equal_pixels = tf.reduce_sum())
    #total_pixels = reduce(lambda x, y: x * y, shape[:3])
    #return equal_pixels / total_pixels
    return 5

def test(load,ckpt_dir):
    batchsize = 4
    imgdir = "DSV"
    groundtruth = "GTV"
    gpu=0.3
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
            print("Restoring", load, ".")
            saver.restore(sess, os.path.join(ckpt_dir,str(load)))
            start = load

        _batch, _labels = prepare_batch(imgdir, groundtruth, batchsize)

        print(_batch.shape)

        # _temp=tf.one_hot(indices=tf.cast(_labels, tf.int32), depth=2)
        #sess.run(logits, feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})
        softmax=sess.run(softmax,feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})

        argmax = tf.argmax(softmax, 3)
        results=np.array(sess.run(argmax))
        print(results)
        print("softmax=", np.array(results).shape)

        accuracy(results,_labels)

        #accuracy()




load=int(sys.argv[1])
ckpt_dir=sys.argv[2]

print("loading", load)
print("loadingdir", ckpt_dir)


test(load,ckpt_dir)

