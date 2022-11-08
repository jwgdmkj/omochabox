# https://www.kaggle.com/code/micajoumathematics/my-first-semantic-segmentation-keras-u-net
#Celebrity Classification
# 1.데이터베이스 2.사용할 네트워크는 안 정해져도 되지만, 후보는 찾아라.(가벼운 것 위주로) 3. 최종적 어플리케이션 데모는?어떻게 데모를 해야 팬시하게 나올지.
import os
import shutil
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras import backend as K

K.set_image_data_format('channels_last')

# 1. 훈련데이터셋 빌딩 --------------------------------------#
datapath = './105_classes_pins_dataset'
trainRatio = 0.8
testRatio = 0.2

dir_list = os.listdir(datapath)
dir_filenum = []
dir_folder = []

for dirname in dir_list:
    if dirname.startswith('pins'):
        #dir_filenum.append(len(os.listdir(datapath + '/' + dirname)))
        dir_folder.append(datapath + '/' + dirname)
#print(dir_folder)
#print(dir_filenum)

# # 80%만 test로 옮기기
# for i, dirname in enumerate(dir_list):
#     if dirname.startswith('pins'):
#         maxidx = int(dir_filenum[i] * testRatio)
#         print(maxidx, end='\t')
#         dir_celeb = os.listdir(datapath+'/'+dirname)
#         for j, file in enumerate(dir_celeb):
#             shutil.move(datapath + '/' + dirname + '/' + file, datapath + '/' + 'test')
#             if j >= maxidx:
#                 break

# 2. 얼굴 검출 --------------------------------------------------
class FaceDetector():

    def __init__(self,faceCascadePath):
        self.faceCascade=cv2.CascadeClassifier(faceCascadePath)


    def detect(self, image, scaleFactor=1.1,
               minNeighbors=5,
               minSize=(30,30)):

        #function return rectangle coordinates of faces for given image
        # scalefactor - 검색 윈도우 확대 비율 (https://darkpgmr.tistory.com/137)

        # minNeighbors - 검출영역으로 선택하기 위한 최소 검출 횟수
        # (검출할 객체 영역에서 얼마나 많은 사각형이 중복되어 검출되어야 최종적으로 객체 영역으로 설정할지)
        # minSize - 검출할 객체의 최소 크기
        rects=self.faceCascade.detectMultiScale(image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects

#Frontal face of haar cascade loaded
frontal_cascade_path='./data/haar_xml/haarcascade_frontalface_default.xml'  #haarcascade-frontal-faces/haarcascade_frontalface_default.xml'

#Detector object created
fd=FaceDetector(frontal_cascade_path)

# ----------------찾아서 보여주기-----------------------#
def getnp(img):
    return np.copy(img)
def show_image(img):
    print(img.shape)
    plt.figure(figsize = (18,15))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces=fd.detect(image_gray,
                   scaleFactor=scaleFactor,
                   minNeighbors=minNeighbors,
                   minSize=minSize)

    for x, y, w, h in faces:
        #detected faces shown in color image
        cv2.rectangle(image,(x,y),(x+w, y+h),(127, 255,0),3)

    show_image(image)
# ----------------찾아서 보여주기 끝-----------------------#


for foldername in dir_folder:
    images = os.listdir(foldername)
    for imagename in images:
        print(foldername + '/' + imagename)
        img = cv2.imread(foldername + '/' + imagename)
        # show_image(getnp(img))

        detect_face(image = getnp(img),
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize = (30,30))

        break
    break


# 3. 훈련 ------------------------------------------------
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

rows, cols = 160, 160 # input과 output에 들어갈 rows, columns
trained_model = MobileNetV2(input_shape=(rows,cols,3),
                    include_top=False,
                    weights='imagenet')
trained_model.trainable=True  #Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.

trained_model.summary()
#print('------------------------------------------------------')
'''
파일 목록을 오름차순으로 정렬하고 이미지 및 마스크로 구분합니다.
각 파일의 형식은 "subject_imageNum.tif" 또는 "subject_imageNum_mask.tif"이므로
정규 표현을 사용하여 각 파일 이름에서 subject 및 imageNum을 추출할 수 있습니다.
"[0-9]+"는 첫 번째 연속 번호를 찾는 것을 의미합니다.
'''
# reg = re.compile("[0-9]+")
# temp1 = list(map(lambda x: reg.match(x).group(), file_list)) # file_list: 해당 디렉토리(train) 내 파일들.
# temp1 = list(map(int, temp1)) # 이를, 이름 순으로 정렬한 다음
# temp2 = list(map(lambda x: reg.match(x.split("_")[1]).group(), file_list)) # _를 구분으로 다시 정렬.
# temp2 = list(map(int, temp2))
#
# # 최종적으로, file_list는 이름으로 잘 정렬된, 즉 1, 10, 11 순이 아닌 1,2,3 순의 (일반-mask)정렬 파일의 list가 됨.
# file_list = [x for _,_,x in sorted(zip(temp1, temp2, file_list))]
# #print(file_list[:20])
#
# train_image = []
# train_mask = []
# # enumerate에 따라, 일반이미지는 image에, mask이미지는 mask리스트에 넣는다.
# for idx, item in enumerate(file_list):
#     if idx %2 == 0:
#         train_image.append(item)
#     else:
#         train_mask.append(item)
#
# # 첫 번째 image와 mask의 first subject를 display
# image1 = np.array(Image.open(path + '1_1.tif'))
# image1_mask = np.array(Image.open(path+'1_1_mask.tif'))
# image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)
#
# fig, ax = plt.subplots(1, 3, figsize = (16,12))
# ax[0].imshow(image1, cmap = 'gray')
# ax[1].imshow(image1_mask, cmap='gray')
# ax[2].imshow(image1, cmap = 'gray', interpolation='none')
# ax[2].imshow(image1_mask, cmap='jet', interpolation='none', alpha = 0.7)
# #plt.show()
#
# # 이러한 image와 mask를 각각 넘파이배열화 시켜 X와 y 배열에 넣는다.
# X, y = [], []
# for image, mask in zip(train_image, train_mask):
#     X.append(np.array(Image.open(path+image)))
#     y.append(np.array(Image.open(path+mask)))
#
# X,y = np.array(X), np.array(y)
# print("X_shape : ", X.shape)
# print("y_shape : ", y.shape)
#
# #---------------2.train_mask 처리----------------------
# mask_df = pd.read_csv('./data/ultrasound-nerve-segmentation/train_masks.csv')
# #print(mask_df.head())
#
# # 첫번째 pixels columns를, mask image로 변환하고자 한다.
# # 사실 이미 mask_image가 제공되기에 이는 불필요한 과정이나.
# # 다른 competition에서 run length encoded data만을 제공하기에, 시험삼아 해본다고 화자가 전함
# width = 580
# height = 420
# temp = mask_df["pixels"][0]
# temp = temp.split(" ")
#
# mask1 = np.zeros(height * width)
# for i, num in enumerate(temp):
#     if i%2 == 0:
#         run = int(num)-1 #very first pixel is 1, not 0
#         length = int(temp[i+1])
#         mask1[run:run+length] = 255
#
# #pixels는 top-bottom으로 번호가 정해지기에(left-right가 아니라)
# mask1 = mask1.reshape((width, height))
# mask1 = mask1.T
# (mask1 != y[0]).sum() #답이 0일 때 잘 나온 것
#
# # RLE : run-length-인코딩
# def RLE_to_image(rle):
#     '''
#     rle: array in mask_df["pixels"]
#     '''
#     width, height = 580, 420
#     if rle == 0:
#         return np.zeros((height, width))
#     else:
#         rle = rle.split(" ")
#         mask = np.zeros(width * height)
#         for i, num in enumerate(rle):
#             if i%2 == 0:
#                 run = int(num) - 1
#                 length = int(rle[i+1])
#                 mask[run:run+length] = 255
#         mask = mask.reshape((width, height))
#         mask = mask.T
#
#         return mask
#
# #----------3. Exploratory data analysis--------------#
# # 우선, train data를 얼마나 가진지 check
# #print("# of train data : ", X.shape[0])
#
# #mask_df.head() : subject, img, pixels가 칼럼
# #reset_index() : 기존 인덱스를 열로 전송하고, 새로운 단순 정수인덱스 세팅
# # 즉, subject_df는 mask_df의 subject, img를 subject기준으로 묶고,
# # 각 subject의 count를 셈
# subject_df = mask_df[['subject', 'img']].groupby(by = 'subject').\
#     agg('count').reset_index()
# subject_df.columns = ['subject', 'N_of_img']
# subject_df.sample(10)
# print(subject_df.sample(10))
# #print(pd.value_counts(subject_df['N_of_img']).reset_index())
#
# '''
# 총 47개의 subjects가 존재하며, 대부분 subject는 120개, 나머지 5개는 119개 존재
# 데이터셋이 similar distribution을 가진지 안가진지 알아보고자 하자.
# train data를 list할 때와 비슷한 방식으로 알 수 있다.
# '''
#
# #---------------4.U-net을만들고, 일단 100개의 data를 이용해 train-----------#
# from keras.models import Model, Input, load_model
# from keras.layers import Input
# from keras.layers.core import Dropout, Lambda
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint
#
# # 암튼 일단 이미지를, train_image와 train_mask에서
# # 일단 랜덤으로 100개만 고른다.
# indices = np.random.choice(range(len(train_image)), \
#     replace = False, size = 100)
# train_image_sample = np.array(train_image)[indices]
# train_mask_sample = np.array(train_mask)[indices]
#
# # Dataset을 빌딩. 일단 X와 y를, 100개짜리 height, width의 empty배열로 놓고, 여기에 넣기 시작.
# IMG_HEIGHT = 96
# IMG_WIDTH = 96
# X = np.empty(shape = (len(indices), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
# y = np.empty(shape = (len(indices), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
# print('x shape: ', end='\t')
# print(X.shape)
# for i, (image_path, mask_path) in enumerate(zip(train_image_sample, train_mask_sample)):
#     image = cv2.imread('./data/ultrasound-nerve-segmentation/train/' + image_path, 0)
#     mask = cv2.imread('./data/ultrasound-nerve-segmentation/train/' + mask_path, 0)
#
#     # 이미지 사이즈 변경, 보간법(interpolation)은 INTER_AREA로, 영상축소시 주로 사용
#     image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
#     mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation =cv2.INTER_AREA)
#
#     X[i], y[i] = image, mask
#
# # 데이터 사이즈의 변경이유 = 우리가 훈련시킬 모델에 입력할 수 있는 차원으로 넣어야 하기에
# # 현재는 (100, height, width)인데, 이를 (100, height, width, 1)로 변경한다(by newaxis)
# # newaxis와 255나누기 통해, [1 4 7]은 [[1/255] [4/255] [7/255]]의 형태가 됨.
# # 이 때, 255로 나누는 것은 0~1사이의 값으로 나누는 것
# X = X[:, :, :, np.newaxis] / 255.
# y = y[:,:,:, np.newaxis] / 255.
# # np,newaxis를 해야, Input((IMG_HEIGHT, IMG_WIDTH, 1))에 맞게 된다.
#
# print("X shape: ", X.shape) # (100,96,96,1)
# print("y shape: ", y.shape) # (100,96,96,1)
#
# # 이제, dice loss와 metrics를 정의하자. 이는 각각 손실함수와 평가지표
# # dice coeff: IoU처럼, 2*|A*B|/(|A| + |B|), 이때 분자와 분모에 smooth가 들어가게됨
# smooth = 1
# def dice_coef(y_true, y_pred): #정답과 예측값을 이용한 계산
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
# def dice_coef_loss(y_true, y_pred): # 손실함수는, metrics(평가지표)의 단순한 음수
#     ret = -dice_coef(y_true, y_pred)
#     return ret
#
# # U-net 모델 설계.
# # inputs = input_size의 shape, 즉 (IMG_HEIGHT, IMG_WIDTH, 1) 사이즈
# # 이 때, inputs 사이즈((IMG_HEIGHT, IMG_WIDTH, 1))에 conv2d를 하면 어떻게 되는가?
# # 이 마지막 1은, 채널이다. 즉, 맨 처음 conv1을 실행하니, 이후 conv1은 ((IMG_HEIGHT, IMG_WIDTH, 32))가 됨.
# inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
#
# # -----------------인코딩----------------------#
# conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
# conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
# pool1 = MaxPooling2D(pool_size= (2,2))(conv1)
#
# conv2 = Conv2D(64, (3,3), activation = 'relu', padding='same')(pool1)
# conv2 = Conv2D(64, (3,3), activation = 'relu', padding='same')(conv2)
# pool2 = MaxPooling2D(pool_size = (2,2))(conv2)
#
# conv3 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool2)
# conv3 = Conv2D(128, (3,3), activation = 'relu', padding='same')(conv3)
# pool3 = MaxPooling2D(pool_size = (2,2))(conv3)
#
# conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
# conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
# conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
# conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#
# # -----------------디코딩----------------------#
# up6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), \
#                                    padding='same')(conv5), conv4], axis=3)
# conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
# conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
#
# up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
# conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
# conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
#
# up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
# conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
# conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
#
# up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
# conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
# conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
#
# # -----------------마지막은 활성화함수--------------------#
# conv10 = Conv2D(1, (1,1), activation = 'sigmoid')(conv9)
#
# model = Model(inputs = [inputs], ouputs = [conv10]) #모델: 들어가는거의 사이즈는 input_size, 나오게하는건 conv10
# model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss,
#               metrics = [dice_coef]) # 손실함수는 dice_coef_loss, 평가함수는 [dice_coef]
# # metrics는, 손실함수와 평가지표를 에포크(epoch)마다 계산한 것을 그려주는데, 손실함수의 추이와 평가지표의 추이를 비교해보면서
# # 모델이 과대적합(overfit) 또는 과소적합(underfit)되고 있는지 여부를 확인
#
# # validation_split : 10%는 test용으로 빼둠.
# results = model.fit(X, y, validation_split = 0.1, batch_size = 4, epochs = 20)
#
#
# #----------------5. Image Generator 선언----------------#
# # Generator 함수 자체는, 위의 100개 데이터 임시테스트에서 쓴 것과 똑같다.
# def Generator(X_list, y_list, batch_size = 16):
#     c = 0
#     while(True):
#         X = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
#         y = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
#
#         for i in range(c, c+ batch_size):
#             image = cv2.imread('./data/ultrasound-nerve-segmentation/train/' \
#                                + X_list[i], 0)
#             mask = cv2.imread('./data/ultrasound-nerve-segmentation/train/' \
#                                + y_list[i], 0)
#             image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), \
#                                interpolation = cv2.INTER_AREA)
#             mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), \
#                               interpolation = cv2.INTER_AREA)
#             X[i-c], y[i-c] = image, mask
#
#         X = X[:,:,:, np.newaxis] / 255
#         y = y[:,:,:, np.newaxis] / 255
#
#         c += batch_size
#         if (c+batch_size >= len(X_list)):
#             c=0
#         yield X, y
#
# X_train, X_val, y_train, y_val = train_test_split(train_image, \
#                                                   train_mask,\
#                                                   test_size = 0.3,\
#                                                   random_state = 1)
# epochs = 10
# batch_size = 8
# steps_per_epochs = int(len(X_train) / batch_size)
# validation_steps = int(len(X_val) / batch_size)
#
# #Generator 함수를 통해, 처음엔 train_gen에선 X를, val_gen에선 y를 반환받는다.
# train_gen = Generator(X_train, y_train, batch_size=batch_size)
# val_gen = Generator(X_val, y_val, batch_size = batch_size)
# model = Model(inputs=[inputs], outputs = [conv10])
# model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss,\
#               metrics = [dice_coef])
#
# history = model.fit_generator(train_gen,  \
#                               steps_per_epochs=steps_per_epochs,\
#                               validation_data = val_gen, \
#                               validation_steps = validation_steps)
#
# #----------------6. dataset 예측------------------------#
# sub = pd.read_csv('./data/ultrasound-nerve-segmentation/sample_submission.csv')
# test_list = os.listdir('./data/test')
#
# print('The # of test data: ', len(test_list))
#
# # test용 데이터를 ascending order로 소팅. 맨 위에서 train에 대해 한 것과 같다.
# reg = re.compile("[0-9]+")
# temp1 = list(map(lambda x: reg.match(x).group(), test_list))
# temp1 = list(map(int, temp1))
# test_list = [x for _, x in sorted(zip(temp1, test_list))]
# test_list[:15]
#
# # test용 빈 np배열 생성
# X_test = np.empty((len(test_list), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
# for i, item in enumerate(test_list):
#     image = cv2.imread('./data/ultrasound-nerve-segmentation/test.'+item, 0)
#     image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation= cv2.INTER_AREA)
#     X_test[i] = image
#
# # X_test를 resize한 후, 255로 정규화시킨 다음, pred 실행
# X_test = X_test[:,:,:, np.axis] / 255
# y_pred = model.predict(X_test)
#
# def run_length_enc(label):
#     from itertools import chain
#     x = label.transpose().flatten()
#     y = np.where(x>0)[0]
#     if len(y) < 10: #consider as empty
#         return ''
#     z = np.where(np.diff(y)>1)[0]
#     start = np.insert(y[z+1], 0, y[0])
#     end = np.append(y[z], y[-1])
#     length = end-start
#     res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
#     res = list(chain.from_iterable(res))
#     return ' '.join([str(r) for r in res])
#
# from skimage.transform import resize
# rles = []
# for i in range(X_test.shape[0]):
#     img = y_pred[i, :, :, 0]
#     img = img > 0.5
#     img = resize(img, (420, 580), preserve_range = True)
#     rle = run_length_enc(img)
#     rles.append(rle)
#     if i%100 == 0:
#         print('{}/{}'.format(i, X_test.shape[0]), end = '\r')
#
# sub['pixels'] = rles
# sub.to_csv('submission.csv', index= False)
