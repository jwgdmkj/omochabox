import os
import cv2

arr = []
arrsum = []
with open('gt.txt', 'r') as file_data:
    lines = file_data.readlines()
    prevnum = 1
    howmany = 0

    img = cv2.imread('./ht21/000001.jpg')
    if img is None:
        print('asdf')

    for i in range(len(lines)):
        parsed = lines[i].split(',')
        curnum = int(parsed[0])
        if int(curnum) > 91 :
            break
        x1, y1, x2, y2 = float(parsed[2]), float(parsed[3]), float(parsed[4]), float(parsed[5])
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        img = cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0,255,0), 1)

        if prevnum != curnum :
            arr.append(howmany)
            if len(arrsum) == 0:
                arrsum.append(howmany)
            else :
                arrsum.append(arrsum[len(arrsum)-1] + howmany)
            howmany = 1
        else :
            howmany += 1

        prevnum = curnum

    # cv2.imshow('rectangle', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print(arr)
print(len(arr))
print(arrsum)
