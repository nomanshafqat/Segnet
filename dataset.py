import os
import cv2
import numpy as np
import random
import tensorflow as tf
import random
from PIL import Image
def prepare_batch(img_dir, ground_truth_dir,batch_size):
    batch =parse(img_dir, ground_truth_dir, batch_size)
    for image,labels in  batch:
        #print(np.array(image).shape)
        img=np.array(image).reshape(batch_size,224,224,3)
        lbel=np.array(labels).reshape(batch_size,224,224)
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

            path = os.path.join(img_dir, filename)
            bgr = cv2.imread(path)
            g = cv2.resize(bgr, (224, 224))
            #g=cv2.divide(g,255)
            groundtruthpath = os.path.join(ground_truth_dir, filename[:-4] + ".png")

            gt = cv2.imread(groundtruthpath, 0)
            gt = cv2.resize(gt, (224, 224))

            ret, thresh1 = cv2.threshold(gt, 100, 255, cv2.THRESH_BINARY)

            thresh1[thresh1>1]=1



            angle=random.randint(0,360)

            g=Image.fromarray(g).rotate(angle)
            thresh1=Image.fromarray(thresh1).rotate(angle)
            g=np.array(g)
            thresh1=np.array(thresh1)


            dataset.append(g)
            labels.append(thresh1)
            #print(filename[:2] , angle,end=" && ")

            #cv2.imshow("g", g)
            #cv2.imshow("gt", thresh1*255)
            #cv2.waitKey(1000)
            if(len(dataset)==batch_size):
                yield dataset,labels
                dataset=[]
                labels=[]
