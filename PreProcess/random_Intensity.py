import cv2
import numpy as np
import random
def intensity_change(image, low, high):

    i_factort=(random.random()*(high-low))+low
    #print("------->Image ",image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    #print(img_hsv)
    img_hsv[:,:,2]=img_hsv[:,:,2]*i_factort
    img_hsv[img_hsv>255]=255
    BGR=cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return  BGR.astype("uint8")