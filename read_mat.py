import cv2
import os, shutil
import scipy.io

filename = 'C:/Users/jsk/PycharmProjects/boostcrowd/Boosting-Crowd-Counting-via-Multifaceted-Attention/datasets/UCF-QNRF_ECCV18/Test/img_0007_ann.mat'
matfile = scipy.io.loadmat(filename)

for i in matfile:
    print(i)

# mat파일 데이터 불러오기
matval = matfile
print("size :", len(matfile['annPoints']))

for i in range(len(matfile['annPoints'])):
    print(matfile['annPoints'][i])

imgname = 'C:/Users/jsk/PycharmProjects/boostcrowd/Boosting-Crowd-Counting-via-Multifaceted-Attention/datasets/UCF-QNRF_ECCV18/Test/img_0007.jpg'
img = cv2.imread(imgname)
img = cv2.circle(img, (95, 894), 30, (0,0,255), -1)
img = cv2.circle(img, (549, 742), 30, (0,0,255), -1)
img = cv2.circle(img, (875, 418), 30, (0,0,255), -1)
cv2.imshow('img', img)
cv2.waitKey()
