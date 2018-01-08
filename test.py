import tensorflow as tf
from Model import SegNet
from dataset import parse,prepare_batch
from loss import loss
import numpy as np
import cv2
def accuracy(sess,logits, labels):
    softmax = tf.nn.softmax(logits)
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

def test():
    batchsize = 4
    imgdir = "/Users/nomanshafqat/Desktop/DIP/IMAGES/DRIVE"
    groundtruth = "GT"
    ckpt_dir = "/Users/nomanshafqat/Desktop/DIP/ckpt/"
    load = 280

    segnet = SegNet(batchsize)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    batch, labels = prepare_batch(imgdir, groundtruth, batchsize)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=1)
    onehot_labels = labels

    logits = segnet.inference(batch)
    print(onehot_labels.get_shape().as_list())

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(init)
        start = 0
        if load > 0:
            print("Restoring", load, ".ckpt.....")
            saver.restore(sess, ckpt_dir + str(load) + ".ckpt")
            start = load

        accuracy(sess,logits,labels)

        #accuracy()






test()