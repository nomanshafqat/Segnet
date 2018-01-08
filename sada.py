import os
import cv2
import numpy as np
import random
import tensorflow as tf
import random
from PIL import Image

def parse(img_dir, ground_truth_dir, batch_size):
    img = os.listdir(img_dir)
    groundtruth = os.listdir(ground_truth_dir)
    # print(img)

    print(img)
    dataset = []
    labels = []
    random.shuffle(img)
    for filename in img:

        if filename.__contains__("DS_Store"):
            continue

        path = os.path.join(img_dir, filename)
        print(path)
        bgr = cv2.imread(path)
        g = cv2.resize(bgr, (224, 224))
        # g=cv2.divide(g,255)
        groundtruthpath = os.path.join(ground_truth_dir, filename[:2] + ".png")
        im=Image.open("/Users/nomanshafqat/Desktop/DIP/IMAGES/GROUNDTRUTH_DRIVE_STARE/DRIVE/"+filename[:2]+"_manual1.gif")
        im.convert('RGB')
        im.save("GT/"+filename[:-4]+".png")
        cv2.imwrite("DS/"+filename[:-4]+".png",bgr)

        '''
        gt = cv2.imread(groundtruthpath, 0)
        gt = cv2.resize(gt, (224, 224))

        ret, thresh1 = cv2.threshold(gt, 100, 255, cv2.THRESH_BINARY)

        thresh1[thresh1 > 1] = 1
        for angle in range(0,360):
        #angle = random.randint(0, 360)

            g = Image.fromarray(g).rotate(angle)
            thresh1 = Image.fromarray(thresh1).rotate(angle)
            g = np.array(g)
            thresh1 = np.array(thresh1)

            dataset.append(g)
            labels.append(thresh1)
            print(filename, angle, end="\t")
            cv2.imwrite("DT/"+filename[:-4]+"_"+str(angle)+".png", g)
            cv2.imwrite("DT/"+filename[:-4]+"_"+str(angle)+".png", thresh1 * 255)

        cv2.imshow("g", g)
        cv2.imshow("gt", thresh1 * 255)
        cv2.waitKey(1000)'''

parse("/Users/nomanshafqat/Desktop/DIP/IMAGES/DRIVE/","GT/",1)