import os
import cv2
import numpy as np
import random
import tensorflow as tf
def prepare_batch(img_dir, ground_truth_dir,batch_size):
    batch =parse(img_dir, ground_truth_dir, batch_size)
    for image,labels in  batch:
        print(np.array(image).shape)
        img=tf.convert_to_tensor(np.array(image).reshape(batch_size,224,224,3),dtype=tf.float32)
        lbel=tf.convert_to_tensor(np.array(labels).reshape(batch_size,224,224,1),dtype=tf.float32)

        return img,lbel

def parse(img_dir, ground_truth_dir,batch_size):
    img = os.listdir(img_dir)
    groundtruth = os.listdir(ground_truth_dir)
    #print(img)


    while True:
        dataset = []
        labels = []
        random.shuffle(img)
        for filename in img:

            if filename.__contains__("DS_Store"):
                continue
            print(filename)

            path = os.path.join(img_dir, filename)
            bgr = cv2.imread(path)
            g = cv2.resize(bgr, (224, 224))

            groundtruthpath = os.path.join(ground_truth_dir, filename[:2] + ".png")

            gt = cv2.imread(groundtruthpath, 0)
            gt = cv2.resize(gt, (224, 224))

            ret, thresh1 = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)


            dataset.append(g)
            labels.append(thresh1)

            if(len(dataset)==batch_size):
                yield dataset,labels
                dataset=[]
                labels=[]
