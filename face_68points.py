import numpy as np
import cv2
import dlib
import os
import random
from PIL import Image

import random
import os

def face_points(sharp_img_path, blur_img_path):
    imgs = sorted(os.listdir(sharp_img_path))
    deference = []


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/asua/Data/Deblur/shape_predictor_68_face_landmarks.dat')

    for ele in imgs:
        img1 = cv2.imread(os.path.join(sharp_img_path, ele))
        img2 = cv2.imread(os.path.join(blur_img_path, ele))

        # 取灰度
        img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # 人脸数rects
        rects1 = detector(img_gray1, 0)
        rects2 = detector(img_gray2, 0)

        #删除数据集中dlib无法识别的人脸
        #if (len(rects1) == 0 or len(rects2) == 0 ):
            #print (ele)
            #os.remove(os.path.join(blur_img_path, ele))
            #os.remove(os.path.join(sharp_img_path, ele))

        for i in range(len(rects1)):
            points1 = np.matrix([[p.x, p.y] for p in predictor(img1, rects1[i]).parts()])

        for i in range(len(rects2)):
            points2 = np.matrix([[p.x, p.y] for p in predictor(img2, rects2[i]).parts()])

            aa = np.sum(np.square(points1 - points2))
            deference.append(np.array(aa))
    return deference

def face_difference(deference, patch_size, batch_size, random_index, step, augment=False):
    img_index = random_index[step * batch_size: (step + 1) * batch_size]

    all_deference = []


    for _index in img_index:
        all_deference.append(deference[_index])

    deference_batch = []

    for i in range(len(all_deference)):

        deference_in = all_deference[i]
        if augment:
            aug = random.randrange(0, 8)
            deference_in = data_augument(deference_in, aug)

        deference_batch.append(deference_in)

    deference_batch = np.array(deference_batch[i])



    return deference_batch

