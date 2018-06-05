import tensorflow as tf
from Model3 import SegNet
from dataset import parse, prepare_batch
from loss import loss
import numpy as np
import cv2
import sys
import os
from  dataset import read_frame


def accuracy(results, labels, _batch, filename):
    print(results.shape, labels.shape, _batch.shape)
    results = np.array(results)
    shape = results.shape

    subtrac = np.subtract(results, labels)

    # print(results)
    # print(labels)

    incoorect = np.count_nonzero(subtrac)
    undetected = len(np.where(subtrac == -1)[0])
    false = len(np.where(subtrac == 1)[0])
    correct = len(np.where(subtrac == 0)[0])

    total = shape[0] * shape[1] * shape[2]

    precision = correct / (correct + false)
    recall = correct / (undetected + correct)
    map = (correct) / (correct + false + undetected)

    print("undetected:", undetected)
    print("false:", false)
    print("correct:", correct)
    print("total:", total)

    print("precision", precision)
    print("recall", recall)
    print("maP", map * 100)

    total = shape[0] * shape[1] * shape[2]

    print("error (%):", (1 - map) * 100)
    # print(results[0]*255)


    cv2.imwrite("results/re_" + filename, np.array(results[0]).astype("uint8") * 255)
    cv2.imwrite("results/gt_" + filename, np.array(labels[0]).astype("uint8") * 255)
    mask = np.expand_dims(np.array(results), axis=-1)
    print(mask[0].shape)
    print(_batch[0].shape)
    np.set_printoptions(threshold=10000)
    results = 1 - results

    # print(_batch[0])
    _batch[0][:, :, 0] = np.multiply(_batch[0][:, :, 0], results[0])

    # print("After=",_batch[0])

    _batch[0][:, :, 1] = np.multiply(_batch[0][:, :, 1], results[0])
    # _batch[0][:, :, 2]=np.multiply(_batch[0][:, :, 2],results[0])

    cv2.imwrite("results/orig_" + filename, _batch[0])

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    # equal_pixels = tf.reduce_sum())
    # total_pixels = reduce(lambda x, y: x * y, shape[:3])
    # return equal_pixels / total_pixels
    return 5


def test(load, ckpt_dir):
    batchsize = 1
    imgdir = "/Users/nomanshafqat/Dropbox/MoreImages/all"
    groundtruth = "/Users/nomanshafqat/val/annotation.json"
    gpu = 0.3
    segnet = SegNet(batchsize)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
    session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    train_batch_plc = tf.placeholder(dtype=tf.float32, shape=[batchsize, 320, 320, 3])
    print(train_batch_plc.get_shape().as_list())

    # labels=tf.one_hot(labels,2)

    logits = segnet.inference(train_batch_plc)

    softmax = tf.nn.softmax(logits)

    print("logits=", logits.get_shape().as_list())

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(init)

        start = 0
        if load > 0:
            print("Restoring", load, ".")
            saver.restore(sess, os.path.join(ckpt_dir, str(load)))
            start = load

        img = os.listdir(imgdir)
        for filename in img:

            if filename.__contains__("DS_Store"):
                continue
            path = os.path.join(imgdir, filename)
            print(path)
            _batch_p = cv2.imread(path)
            _batch_p=cv2.GaussianBlur(_batch_p, (17,17), 40.0)

            _batch = cv2.resize(_batch_p, (320, 320))

            _batch_ex = np.expand_dims(_batch, axis=0)
            print(_batch.shape)

            # _temp=tf.one_hot(indices=tf.cast(_labels, tf.int32), depth=2)
            # sess.run(logits, feed_dict={train_batch: _batch.astype(np.float32), labels: _labels})
            softmaxa = sess.run(softmax, feed_dict={train_batch_plc: _batch_ex.astype(np.float32)})

            argmax = tf.squeeze(tf.argmax(softmaxa, -1), axis=0)
            results = np.array(sess.run(argmax))
            print(results)
            print("softmax=", np.array(results).shape)
            print(_batch.shape)
            mask = cv2.resize(results, _batch.shape[:-1])
            _batch[:, :, 0] = _batch[:, :, 0] * (1 - mask)
            _batch[:, :, 1] = _batch[:, :, 1] * (1 - mask)

            #_batch_p [i:i+a,j:j+b,:]= _batch

            # cv2.imwrite("results/"+filename+"_re.jpg",mask)
            cv2.imwrite("results/" + filename +".jpg", _batch)

            # accuracy(results,_labels,_batch,filename)

            # accuracy()


load = int(sys.argv[1])
ckpt_dir = sys.argv[2]

print("loading", load)
print("loadingdir", ckpt_dir)

test(load, ckpt_dir)
