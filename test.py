import tensorflow as tf
from model3 import SegNet
from dataset import parse,prepare_batch
from loss import loss
import numpy as np
import cv2
import sys
import os
def accuracy(results ,labels,filename):


    results=np.array(results)
    shape=results.shape


    subtrac=np.subtract(results,labels)


    #print(results)
    #print(labels)

    incoorect=np.count_nonzero(subtrac)
    undetected=len(np.where(subtrac==-1)[0])
    false=len(np.where(subtrac==1)[0])
    correct=len(np.where(subtrac==0)[0])

    total=shape[0]*shape[1]*shape[2]

    precision=correct/(correct+false)
    recall=correct/(undetected+correct)
    map=(correct ) /  (correct+ false + undetected)


    print("undetected:",undetected)
    print("false:",false)
    print("correct:",correct)
    print("total:",total)

    print("precision",precision)
    print("recall",recall)
    print("maP",map*100)

    total=shape[0]*shape[1]*shape[2]


    print("error (%):",(1-map)*100)
    print(results[0]*255)


    cv2.imwrite("results/re_"+filename,np.array(results[0]).astype("uint8")*255)
    cv2.imwrite("results/gt_"+filename,np.array(labels[0]).astype("uint8")*255)



    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    #equal_pixels = tf.reduce_sum())
    #total_pixels = reduce(lambda x, y: x * y, shape[:3])
    #return equal_pixels / total_pixels
    return 5

def test(load,ckpt_dir):
    batchsize = 1
    imgdir = "DSV"
    groundtruth = "GTV"
    gpu=0.3
    segnet = SegNet(batchsize)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch = tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])
    labels = tf.placeholder(dtype=tf.int32, shape=[batchsize, 512, 512])
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

        img = os.listdir(imgdir)
        for filename in img:

            if filename.__contains__("DS_Store"):
                continue
            inputwidth = 512
            path = os.path.join(imgdir, filename)
            g = cv2.imread(path)
            g = cv2.resize(g, (512, 512))

            groundtruthpath = os.path.join(groundtruth, filename[:-4] + ".png")

            gt = cv2.imread(groundtruthpath, 0)
            gt = cv2.resize(gt, (512, 512))

            ret, thresh1 = cv2.threshold(gt, 100, 255, cv2.THRESH_BINARY)
            # square=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            # thresh1=morphology.dilation(thresh1,square)
            thresh1[thresh1 > 1] = 1
            _batch=np.array([g])
            _labels=np.array([thresh1])

            print(_batch.shape)

            # _temp=tf.one_hot(indices=tf.cast(_labels, tf.int32), depth=2)
            #sess.run(logits, feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})
            softmaxa=sess.run(softmax,feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})

            argmax = tf.argmax(softmaxa, 3)
            results=np.array(sess.run(argmax))
            print(results)
            print("softmax=", np.array(results).shape)

            accuracy(results,_labels,filename)

        #accuracy()




load=int(sys.argv[1])
ckpt_dir=sys.argv[2]

print("loading", load)
print("loadingdir", ckpt_dir)


test(load,ckpt_dir)

