import os,cv2,random

def prepare(img_dir, ground_truth_dir):
    img = os.listdir(img_dir)
    groundtruth = os.listdir(ground_truth_dir)
    #print(img)




    for filename in img:

        if filename.__contains__("DS_Store"):
            continue
        path = os.path.join(img_dir, filename)
        o = cv2.imread(path)
        #g = cv2.resize(g, (300, 300))
        #g=cv2.divide(g,255)
        groundtruthpath = os.path.join(ground_truth_dir, filename[:6] + "-vessels4.ppm")
        g = cv2.imread(groundtruthpath)


        cv2.imwrite("GT/"+filename[:-3]+"png",g)
        cv2.imwrite("DS/"+filename[:-3]+"png",o)
        #cv2.waitKey(0)


prepare("/Users/nomanshafqat/Desktop/DIP/IMAGES/STARE","/Users/nomanshafqat/Desktop/DIP/IMAGES/GROUNDTRUTH_DRIVE_STARE/Hoover")