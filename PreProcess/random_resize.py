import random
import cv2
import tensorflow as tf


def random_resize(img, gt):
    w = img.shape[0]
    h = img.shape[1]

    a = random.randint(650, w)
    b = random.randint(650, h)

    img = cv2.resize(img, (a, b))
    gt = cv2.resize(gt, (a, b))
    return img, gt
