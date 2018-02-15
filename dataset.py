import os
import cv2
import numpy as np
import random
import tensorflow as tf
import random
from PIL import Image
from skimage import morphology
import xml.etree.ElementTree as ET
from PreProcess.random_crop import random_crop
from PreProcess.random_resize import random_resize
from PreProcess.random_Intensity import intensity_change
from PreProcess.random_rotate import randomrotate

def read_frame(filname, img_path="", ann_path=""):
    img_path = os.path.join(img_path, filname)
    ann_path = os.path.join(ann_path, filname[:-3] + "xml")

    img = cv2.imread(img_path)

    labels = np.zeros((np.array(img).shape[:-1]))

    # print(labels)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    # actual parsing
    in_file = open(ann_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    jpg = str(root.find('filename').text)
    imsize = root.find('size')
    w = int(imsize.find('width').text)
    h = int(imsize.find('height').text)
    all = list()
    dumps = list()

    for obj in root.iter('object'):
        current = list()
        name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))
        current = [name, xn, yn, xx, yx]
        labels[yn + 5:yx - 5, xn + 3:xx - 3] = 1
        all += [current]

    add = [[jpg, [w, h, all]]]
    in_file.close()
    # print(add)
    return img, labels


def fetch_data(batch_size, img_dir, ann_dir):
    batch_labels = []
    batch_img = []
    filenames = os.listdir(img_dir)
    while True:
        random.shuffle(filenames)
        for filename in filenames:

            if not filename.__contains__("jpg"):
                continue

            print(filename, end=" ")
            img, labels = read_frame(filename, img_dir, ann_dir)

            batch_img.append(img)

            if len(batch_img) == batch_size:
                yield batch_img, batch_labels
                batch_labels = []
                batch_img = []


def prepare_batch(img_dir, ground_truth_dir, batch_size):
    batch = parse(img_dir, ground_truth_dir, batch_size)
    for image, labels in batch:
        # print(np.array(image).shape)
        img = np.array(image).reshape(batch_size, 512, 512, 3)
        lbel = np.array(labels).reshape(batch_size, 512, 512)
        return img, lbel


def parse(img_dir, ground_truth_dir, batch_size):
    img = os.listdir(img_dir)
    groundtruth = os.listdir(ground_truth_dir)
    # print(img)


    while True:

        dataset = []
        labels = []
        random.shuffle(img)
        for filename in img:

            if not filename.__contains__("jpg"):
                continue
            inputwidth = 512

            img, gt = read_frame(filename, img_dir, ground_truth_dir)

            img, gt = random_resize(img, gt)

            img, gt = random_crop(img, gt)
            img, gt = randomrotate(img, gt)
            #print(img)
            #cv2.imshow("img1", img)
            #print(img)

            img = intensity_change(img, 0.5, 1.5)
            #cv2.imshow("img", img)
            #cv2.imshow("gt", gt)
            #cv2.waitKey(5000)
            '''
            ret, thresh1 = cv2.threshold(gt, 100, 255, cv2.THRESH_BINARY)
            #square=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            #thresh1=morphology.dilation(thresh1,square)
            thresh1[thresh1>1]=1

            #tf.random_resize(g,[224,224,3],tf.set_random_seed(random.randint(0,99999)))
            #angle=random.randint(0,360)
            #g=Image.fromarray(g).rotate(angle)
            #thresh1=Image.fromarray(thresh1).rotate(angle)

            g=np.array(g)
            thresh1=np.array(thresh1)

            h=g.shape[0]-inputwidth
            w=g.shape[1]-inputwidth
            #print(h,w,g.shape)
            hoff=random.randint(0,h)
            woff=random.randint(0,w)

            g=g[hoff:hoff+inputwidth,woff:woff+inputwidth,:]
            thresh1=thresh1[hoff:hoff+inputwidth,woff:woff+inputwidth]
            for angle in range(0,360,90):
                img = Image.fromarray(img).rotate(angle)
                gt = Image.fromarray(gt).rotate(angle)

                g = np.array(img)
                thresh1 = np.array(img)'''

            dataset.append(img)
            labels.append(gt)

            # cv2.imshow("g", g)
            # cv2.imshow("gt", thresh1*255)
            # cv2.waitKey(1000)
            if (len(dataset) == batch_size):
                # print(np.array(dataset).shape)

                yield np.array(dataset), np.array(labels)
                dataset = []
                labels = []
