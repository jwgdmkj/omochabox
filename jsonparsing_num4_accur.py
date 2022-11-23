import json
import cv2
"""
숫자4개랑 score 있는 json txt로 변환한 다음
이미지에 사각형으로 그리는거
"""
obj = open('coco_instances_results.json')
jsonstring = json.load(obj)

with open('./firstimage.txt', "w") as f:
    for idx in jsonstring:
        if idx['image_id'] == 1:
            inputstr = str(round(idx['bbox'][0])) + ' ' + str(round(idx['bbox'][1])) + ' ' + \
                       str(round(idx['bbox'][2])) + ' ' + str(round(idx['bbox'][3])) + ' ' + str(idx['score']) + '\n'
            # print(str(inputstr))
            f.write(inputstr)

with open('./secondimage.txt', "w") as f:
    for idx in jsonstring:
        if idx['image_id'] == 2:
            inputstr = str(round(idx['bbox'][0])) + ' ' + str(round(idx['bbox'][1])) + ' ' + str(round(idx['bbox'][2])) + ' ' + \
                       str(round(idx['bbox'][3])) + ' ' + str(idx['score']) + '\n'
            # print(str(inputstr))
            f.write(inputstr)

with open('./thirdimage.txt', "w") as f:
    for idx in jsonstring:
        if idx['image_id'] == 3:
            inputstr = str(round(idx['bbox'][0])) + ' ' + str(round(idx['bbox'][1])) + ' ' + str(round(idx['bbox'][2])) + ' ' + \
                       str(round(idx['bbox'][3])) + ' ' + str(idx['score']) + '\n'
            # print(str(inputstr))
            f.write(inputstr)


with open('./firstimage.txt', "r") as f:
    img = cv2.imread('./joongangtest.jpg')
    idx = f.readlines()
    for i in idx:
        arr = i.split()
        if float(arr[4]) > 0.5:
            cv2.rectangle(img, (int(arr[0]), int(arr[1])), (int(arr[2]), int(arr[3])), (0, 255, 0), 1)

    cv2.imshow('rec', img)
    cv2.imwrite('./joongangtst_result.jpg', img)

with open('./secondimage.txt', "r") as f:
    img = cv2.imread('./joongangtest_up.jpg')
    idx = f.readlines()
    for i in idx:
        arr = i.split()
        if float(arr[4]) > 0.5:
            cv2.rectangle(img, (int(arr[0]), int(arr[1])), (int(arr[2]), int(arr[3])), (0, 255, 0), 1)

    cv2.imshow('rec', img)
    cv2.imwrite('./joongangtst_up_result.jpg', img)

with open('./thirdimage.txt', "r") as f:
    img = cv2.imread('./humansample.jpg')
    idx = f.readlines()
    for i in idx:
        arr = i.split()
        if float(arr[4]) > 0.975:
            cv2.rectangle(img, (int(arr[0]), int(arr[1])), (int(arr[2]), int(arr[3])), (0, 255, 0), 1)

    cv2.imshow('rec', img)
    cv2.waitKey(0)
    cv2.imwrite('./humansample_result.jpg', img)
