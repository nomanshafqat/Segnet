import math
import random

def random_crop(img,labels):

    w=img.shape[0]
    h=img.shape[1]
    a=random.randint(0,w-512)
    b=random.randint(0,h-512)

    img=img[a:a+512,b:b+512]
    labels=labels[a:a+512,b:b+512]

    return img,labels


