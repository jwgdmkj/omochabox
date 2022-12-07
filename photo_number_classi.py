"""
숫자 각각에 설정을 할당해,
예를 들어 1번이면 해당 사진은 낮에 찍은 사진
2번이면 밤에 찍은 사진 등을 분간하는 코드
"""
import sys
import cv2
import os, shutil
import scipy.io
from PIL import Image
# base = 'C:/Users/jsk/Desktop/김정수/빅컴_이미지'
# base = os.path.join('\\\\mldisk.sogang.ac.kr', 'exchange', '이원일', '빅컴_이미지')

base = './bigcom_img'
baselist = os.listdir(base)
day, night = 0, 0
occlusion = 0

# for i in baselist:
#     if i.endswith('db') or i.endswith('txt') or i.endswith('json'):
#         continue
#
#     if i.endswith('.jpg') or i.endswith('.JPG'):
#         appendix = i[-11:]
#     elif i.endswith('.JPEG') or i.endswith('.jpeg'):
#         appendix = i[-12:]
#     print(i)
#     if(i[3] == '가'):
#         #print(base,'/', i)
#         #print(appendix)
#         os.rename(base+'/'+ i, base + '/horizon'+appendix)
#     elif(i[3]=='세'):
#         #print(base, '/', i)
#         #print(appendix)
#         os.rename(base + '/' + i, base + '/vertical' + appendix)
#     else:
#         print(base, '/', i)

# day : 1 / night : 2 / day+occlusion : 3 / night + occlusion : 4
for idx in range(len(baselist)):
    if not baselist[idx].endswith('.jpg') and not baselist[idx].endswith('.JPG') and not baselist[idx].endswith('.jpeg')\
            and not baselist[idx].endswith('.JPEG'):
        continue

    print(base + '/' + baselist[idx])
    # 이미지 보여주고, waitkey함.
    img = cv2.imread(base + '/' + baselist[idx])
    if img is None:
        print('no img')
        sys.exit()

    cv2.imshow('img', img)

    key = cv2.waitKey()
    if key == ord('1'):
        day += 1
    elif key == ord('2'):
        night += 1
    elif key == ord('3'):
        day += 1
        occlusion += 1
    elif key == ord('4'):
        night += 1
        occlusion += 1
    cv2.destroyAllWindows()
    print('현 ', idx, '번째 ', '파일명 ' +  baselist[idx], 'day: ',day, 'night: ', night, 'occlu: ', occlusion)
