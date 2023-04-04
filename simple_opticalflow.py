import numpy as np
import cv2
import scipy.ndimage
import json
import os
import matplotlib.pyplot as plt

# 1) density map 넣어보기 --> 성능 나옴. sigma를 다르게 해보면?
# 2) 걍 gt box 때려넣어서 연결하기
# 3) input으로 gt box 넣어보기 --> gt_box의 중심점을 prevPt로 넣으면, 에러 발생
# (error: (-215:Assertion failed) (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 in function 'cv::`anonymous-namespace'::SparsePyrLKOpticalFlowImpl::calc')
import torch

""" make video into img
filepath = './Stempede_2.mp4'
vidcap = cv2.VideoCapture(filepath) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

success,image = vidcap.read()
count = 0
while success:
    filename = str(count).zfill(5) + '.jpg'
    cv2.imwrite("./Stempede_2/" + filename, image)     # save frame as JPEG file
    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1

print("finish! convert video to frame")
"""

"""   # ht21 dataset
gt_arr = []
arr = []
width, height = 1920, 1080
with open('gt.txt', 'r') as file_data:
    lines = file_data.readlines()
    prevnum = 1
    howmany = 0

    for i in range(len(lines)):
        parsed = lines[i].split(',')
        curnum = int(parsed[0])
        if int(curnum) > 61 :
            break

        x1, y1, x2, y2 = float(parsed[2]), float(parsed[3]), float(parsed[4]), float(parsed[5])
        # x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        x, y = round(x1) + round(x2/2), round(y1) + round(y2/2)

        if prevnum != curnum :
            gt_arr.append(arr)
            arr = [[curnum, y, x]]
        else:
            arr.append([curnum, y, x])

        prevnum = curnum

print(gt_arr)
"""
"""   # peopleflow dataset
arr = []
arrsum = []
width, height = 1920, 1080
json_arr = []
gt_arr = []
for jsonfile in os.listdir('./people_flow'):
    if jsonfile.endswith('.json'):
        json_arr.append(jsonfile)

for gtidx in range(len(json_arr)):
    with open(os.path.join('./people_flow', json_arr[gtidx]),'r') as f:
        gt = json.load(f)
    anno_list = list(gt.values())[0]['regions']
    tmparr = []
    for i in range(0, len(anno_list)):
        y_anno = int(anno_list[i]['shape_attributes']['y'])
        x_anno = int(anno_list[i]['shape_attributes']['x'])
        tmparr.append([x_anno, y_anno])
    gt_arr.append(tmparr)
    if gtidx >= 60 :
        break
first_gt = []
for i in range(len(gt_arr[0])):
    first_gt.append([gt_arr[0][i]])
"""
""" # make gaussian filter
width, height = 1920, 1080
img_size = (width, height)
imgidx = 0
for gt in gt_arr:
    density_map = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
    # print(gt)
    # add points onto basemap
    for point in gt:
        base_map = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
        # subtract 1 to account for 0 indexing
        base_map[int(round(point[1]) - 1), int(round(point[2]) - 1)] += 1
        density_map += scipy.ndimage.filters.gaussian_filter(base_map, sigma=16, mode='constant')

    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='hot', interpolation='nearest')
    # plt.show()
    ax.set_axis_off()
    plt.savefig('./density_map/ht21_large_' + str(imgidx).zfill(3) + '.png', bbox_inches='tight', dpi = 300)
    plt.close(fig)
    print(imgidx)
    imgidx += 1
"""
"""
pathIn= './people_flow_optical'
pathOut = './people_flow_optical.mp4'
fps = 30
paths = os.listdir(pathIn)
frame_array = []
for idx , path in enumerate(paths) :
    imgpath = pathIn + '/' + path
    if not imgpath.endswith('png'):
        continue
    img = cv2.imread(imgpath)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, (960, 540))
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
"""
"""
cap = cv2.VideoCapture('./people_flow_optical.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
print(fps)
delay = int(1000/fps)
# 추적 경로를 그리기 위한 랜덤 색상
color = np.random.randint(0,255,(200,3))
lines = None  #추적 선을 그릴 이미지 저장 변수
prevImg = None  # 이전 프레임 저장 변수
# calcOpticalFlowPyrLK 중지 요건 설정
termcriteria =  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# first_gt = np.array(first_gt)
pathOut = './people_flow_optical_flow.mp4'
framearray = []
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, (1920, 1080))

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우
    if prevImg is None:
        prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출  ---①
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)    # [point num, 1, 2], ndarray
        # prevPt = first_gt

        # imgg = cv2.imread('./people_flow/001.jpg')
        # for i in range(len(prevPt)):
        #     cv2.circle(imgg, (int(prevPt[i][0][0]), int(prevPt[i][0][1])), 10, (0, 0, 255), 5)
        # cv2.imshow('OpticalFlow-LK', imgg)
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        nextImg = gray
        # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
        # we pass the previous frame, previous points and next frame.
        # It returns next points along with some status numbers which has a value of 1 if next point is found, else zero.
        # criteria : 반복 알고리즘의 종료 기준 / maxLevel : 최대 피라미드 레벨 / winSize : 각 피라미드 레벨에서 검색할 윈도우 크기(기본은 (21, 21))
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, \
                                        prevPt, None, criteria=termcriteria)

        # 대응점이 있는 코너, 움직인 코너 선별 ---③
        prevMv = prevPt[status==1]
        nextMv = nextPt[status==1]


        for i,(p, n) in enumerate(zip(prevMv, nextMv)):
            px,py = p.ravel()
            nx,ny = n.ravel()

            px, py = round(px), round(py)
            nx, ny = round(nx), round(ny)

            # 이전 코너와 새로운 코너에 선그리기 ---④
            cv2.line(lines, (px, py), (nx,ny), color[i].tolist(), 2)

            # 새로운 코너에 점 그리기
            cv2.circle(img_draw, (nx,ny), 2, color[i].tolist(), -1)

        # 누적된 추적 선을 출력 이미지에 합성 ---⑤
        img_draw = cv2.add(img_draw, lines)
        # 다음 프레임을 위한 프레임과 코너점 이월
        prevImg = nextImg
        prevPt = nextMv.reshape(-1,1,2)

    cv2.imshow('OpticalFlow-LK', img_draw)
    framearray.append(img_draw)
    key = cv2.waitKey(delay)
    if key == 27 : # Esc:종료
        break
    elif key == 8: # Backspace:추적 이력 지우기
        prevImg = None
cv2.destroyAllWindows()
cap.release()
for i in range(len(framearray)):
    # writing to a image array
    # resize_frame = cv2.resize(framearray[i], (514, 298))
    out.write(framearray[i])
    cv2.imshow('asdf', framearray[i])
out.release()
"""

# What to make in AdvancedAlgorithm Class
# Head 좌표를 받아, 그 좌표를 선으로 잇는 이미지 생성
# 이 점을 일정시간 유지시킨 다음 지워야 하기 때문에, 한 프레임에서 잡힌 헤드들은 전부 time을 갖고 있어야 함
# 아마 큐 시스템을 적용해서(약 30) 30프레임 이상 유지된 것들은 pop시키는 편이 나을듯
# 위의 kanade 알고리즘처럼, 따로 캔버스를 만들어 거기에 선을 그린 다음, 본래 이미지에 add시키면 시각화에 좋다.
# 자기랑 가장 가까운 녀석을 찾아, 선으로 잇도록 하자.

"""
from glob import glob
imgdir = sorted(glob(os.path.join('./density_map', 'flow*')))
lines = np.zeros((1080, 1920))
color = np.random.randint(0, 255, (200, 3))
print(gt_arr)
for i in range(len(gt_arr)):
    if i == 0 :
        prevpt = gt_arr[0]
    else :
        # 모든 한 장면에 gt에 대해 최근접한 녀석 찾기
        for j in range(len(gt_arr[i])):
            matchpoint = None
            distance = 210000000
            for k in range(len(prevpt)):
                if
            cv2.circle(lines, (gt_arr[i][j][0], gt_arr[i][j][1]), 2, color[j].tolist(), -1)
            # img_draw = cv2.add(img_draw, lines)
        cv2.imshow('lines', lines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        prevpt = gt_arr[i]  # prevpt 갱신
"""
