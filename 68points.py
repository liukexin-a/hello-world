import numpy
import dlib
import cv2
import sys
import os

#dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
dirname, filename = os.path.split(sys.argv[0])
# print(dirname,filename)
# path = os.getcwd()
os.chdir(dirname)
print(os.getcwd())

# 人脸检测
detector = dlib.get_frontal_face_detector()

# 人脸关键点标注。
predictor = dlib.shape_predictor('/home/asua/Data/Deblur/shape_predictor_68_face_landmarks.dat')
img = cv2.imread('/home/asua/Data/Deblur/Richard_E_Grant_48376.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#这样也可以灰度图但是不建议用：im2 = cv2.imread('tfboys.jpg',flags = 0)

dets = detector(gray,0)# 第二个参数越大，代表讲原图放大多少倍在进行检测，提高小人脸的检测效果。

for d in dets:
 # 使用predictor进行人脸关键点检测 shape为返回的结果
    shape = predictor(gray, d)
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 1, (255, 0, 0), 2)
        #利用cv2.putText标注序号
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(index+1),pt_pos,font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()