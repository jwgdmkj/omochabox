# 1. 훈련데이터셋 빌딩 --------------------------------------#
"""
해당 프로그램의 목표 :
어떠한 이미지 세트가 있고 
하나의 텍스트파일에 그 이미지에 대한 '얼굴개수' 와 '얼굴좌표'가 들어있을 때,
이를 각 이미지당 하나의 txt를 만들고, 거기의 얼굴개수와 정규화시킨 얼굴좌표를 집어넣는 프로그램 
"""

import os
import sys
import shutil
from PIL import Image
# import cv2 as cv

# image = cv2.imread('C:/Users/jsk/PycharmProjects/0_Parade_marchingband_1_353.jpg')
# if image is None:
#     print('Image load failed')
#     sys.exit()
#
# red_color = (0,0,255)
# image = cv2.rectangle(image, (263, 381), (263+113, 381+169), red_color, 3)
# image = cv2.rectangle(image, (635, 271), (635+134, 271+169), red_color, 3)
# cv2.imshow('rec', image)
# cv2.waitKey()

if True:
    # 텍스트파일을 읽어서, jpg로 끝나는 게 있다면 그 파일을 찾은 다음 그 다음줄의 수(사람 수)만큼 for문을 돌린다. 해당 이미지의 사이즈를 얻어와, 1로 만들고 그 몫만큼 해당 숫자를 나눠준다(앞 4개만)
    textpath = 'C:/Users/jsk/Downloads/wider_face_split/wider_face_split'
    imgpath = 'C:/Users/jsk/PycharmProjects/dataset/images/val/'
    labelpath = 'C:/Users/jsk/PycharmProjects/dataset/labels/val'
    dir_list = os.listdir(textpath)

    for file in dir_list:
        # txt파일인 경우엔, 그 파일을 읽어와서, 파일_new 생성 후, 여기에 한줄씩 읽어들여 jpg로 끝나면 / 앞부분을 제거하고, 아니면 그대로 쓸 예정임.
        if file.endswith('.txt') and file.startswith('wider_face_val') :
            with open(textpath + '/' + file, "r") as f: # 파일의 line를 읽어들여
                lines = f.readlines()
                i = 0
                print(len(lines))
                while i < len(lines):  # 그 line 수만큼 진행할 거임
                    tmparr = lines[i].split('/')  # / 기준으로 파싱해서
                    # print(labelpath + '/' + tmparr[1][:-5])
                    with open(labelpath + '/' + tmparr[1][:-5] + '.txt', "w") as nf: # 파싱한 것의 오른쪽같은 이름의 txt파일을 생성해
                        image = Image.open(imgpath + tmparr[1][:-1]) # 그 이미지 읽어들임
                        h, w = image.size

                        loopnum = int(lines[i+1])  # 해당 이미지에 있는 얼굴 개수. 이거도 작성해야 됨.
                        nf.write(str(loopnum))

                        # 해당 공간에 얼굴이 0개라면, 0 0 0 0 만 저장되어 있으며, 여길 더 건너뛰어야 됨.
                        if loopnum == 0:
                            i += 1

                        nf.write('\n')
                        for j in range(loopnum):
                            points = lines[i+2+j].split()  # 해당 이미지의 j번째 얼굴의 숫자들을 리스트로

                            # 이미지는 y시작, x시작, y길이, x길이로 되어있음. 이 points들을 작성해야 됨.
                            points[0] = str(int(points[0])/h)
                            points[1] = str(int(points[1])/w)
                            points[2] = str(int(points[2])/h)
                            points[3] = str(int(points[3])/w)

                            tmpstr = ''
                            for k in range(len(points)):
                                tmpstr += points[k]
                                tmpstr += ' '

                            tmpstr += '\n'
                            # print(tmpstr)
                            nf.write(tmpstr)

                    i = i + loopnum + 1 + 1   # 기존 i + 얼굴갯수 있는 줄 + 얼굴갯수의 다음줄로 넘어가야 됨.
    
    # dir_filenum = []
    # dir_folder = []
    # print('dirlist ', dir_list)
    # for dirname in dir_list:
    #     dir_folder.append(datapath + '/' + dirname)
    # print('dirfolder : ' , dir_folder)
    # print('dirnum: ' , dir_filenum)
    #
    # move_dir = '//mldisk.sogang.ac.kr/exchange/김정수/Face_detection/WIDER_FACE/WIDER_val/all_img'
    # # os.makedirs(datapath + '/' + 'train')
    # # os.makedirs(datapath + '/' + 'test')
    #
    # # 80%만 test로 옮기기
    # for i, dirname in enumerate(dir_list):
    #     dir_celeb = os.listdir(datapath+'/'+dirname)
    #     for j, file in enumerate(dir_celeb):
    #         #print(datapath + '/' + dirname + '/' + file)
    #         if(file == 'Thumbs.db'):
    #             continue
    #         shutil.move(datapath + '/' + dirname + '/' + file, move_dir)r
