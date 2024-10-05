import numpy as np
import cv2
import os


def motion_blur(image, degree=50, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred




if __name__ == '__main__':
    folder = '/home/asua/Data/Deblur/data/train/1/face'
    folder_to_save = '/home/asua/Data/Deblur/data/train/1/face_cv'

    for path in os.listdir(folder):
        print(path)
        img = cv2.imread(os.path.join(folder, path))
        img = motion_blur(img)
        cv2.imwrite(os.path.join(folder_to_save, path), img)