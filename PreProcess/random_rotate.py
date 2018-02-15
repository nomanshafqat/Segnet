import cv2
import random
from PIL import Image
import numpy as np
def randomrotate(img,labels):
    angle=90*int(random.random()*4)
    print(angle)
    cv2.imshow("asvwdas",img)

    cv2.rotate(img,angle)
    img=np.array(Image.fromarray(img).rotate(angle))
    labels=np.array(Image.fromarray(labels).rotate(angle))
    cv2.imshow("asdas",img)
    return np.array(img), np.array(labels)

