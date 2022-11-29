import json
import cv2
import os, shutil

"""
test와 train의 각 jpg와 ann으로 끝나는 것들을
각자 옮겨줄거임
"""

test_path = 'C:/Users/jsk/PycharmProjects/lsc-cnn/lsc-cnn/UCF-QNRF/test_data/ground_truth'
train_path = 'C:/Users/jsk/PycharmProjects/lsc-cnn/lsc-cnn/UCF-QNRF/train_data/ground_truth'

content = '0.033917777 0.055902213 0.22611132 0.08526811 0.0 0.0 0.0 0.0 0.0 0.0 0.018562682 0.046558548 0.11408186 0.1830801 0.03981372 0.0021608993 0.0 0.0 0.0 0.0 0.0063239187 0.05196043 0.08898266 0.19097553 0.05260977 0.010188252 0.0 0.0 0.0 0.0 0.017614916 0.040981628 0.025714118 0.0 0.0 0.0 0.0 0.0 0.021236032 0.0799488 0.0747301 0.017062664 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.046599913 0.10653201 0.016422644 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0014377534 0.032022856 0.0031697676 0.0 0.0019863844 0.15994668 0.31328055 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.06283361 0.25549805 0.034382395 0.0 0.0 0.0 0.0 0.0 0.0 0.013456583 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.06469628 0.2289798 0.021121085 0.0 0.0 0.0 0.0 0.0 0.028443038 0.3149817 0.108737454 0.012295626 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.19503875 0.36584148 0.03724973 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.11927748 0.019831285 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.005976051 0.0 0.0 0.0 0.0 0.0 0.0 0.010391951 0.32518816 0.16621712 0.0 0.0 0.0 0.0 0.0 0.0 0.020025536 0.0060237944 0.0 0.0 0.0 0.060632616 \
0.0 0.0 0.0 0.0 0.013410002 0.025910247 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.06204231 0.45289305 0.16075261 0.017351225 0.0 0.0 0.0 0.020701937 0.115159124 0.023088336 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01274395 0.01912883 0.0 0.0 0.0 0.0 0.0 0.14624904 0.49412945 0.0583506 0.0 0.0 0.026922747 0.22385967 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.008837558 0.123465255 0.036909357 0.0 0.0 0.0 0.0 0.14867006 0.47196525 0.08212664 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.037190933 0.2248186 0.022560164 0.0 0.0 0.007671699 0.1190611 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.020894855 0.1049287 0.028345589 0.0 0.0 0.0 0.0 0.0 0.05710703 0.14017001 0.025208216 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.034254245 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.23312981 0.42188102 0.038042977 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.020679705 0.26070076 0.036247913 0.0 0.0 0.0 0.08736938 0.08937841 0.015754864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.031174514 0.059454124 0.0029962882 0.0 0.0 0.0 0.0 0.0 0.003498651 0.17093581 0.11446378 0.012027875 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.01265347 0.0 0.0 0.0 0.0 0.069276094 0.45421487 0.10109412 0.005824223 0.0 0.0033947676 0.31816688 0.36727503 0.03195638 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.011454999 0.3398967 0.27810055 0.025541037 0.0 0.0 0.0 0.0 0.0 0.0 \
0.15363002 0.029664092 0.0 0.0 0.0 0.0 0.0 0.033318784 0.014086232 0.0 0.0 0.0 0.03867036 0.049703844 0.003158316 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.032713406 0.02783572 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.30071142 0.059221994 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.021573156 0.17577799 0.031788643 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.084914476 0.005563125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.12908494 0.43189242 0.09043269 0.009606399 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.17764151 0.1871387 0.004382625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.006779805 0.07383588 0.02565964 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.28579354 0.30442286 0.016659066 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015481338 0.018578045 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.014930084 0.012367897 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07367235 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07692638 0.47169277 0.088054165 0.0013561398 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07369886 0.18813524 0.013734557 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19181095 0.27691668 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.035074417 0.2515775 0.040946193 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.23324874 0.38122115 0.037847992 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012787484 0.049530093 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.023183875 0.056473456 0.0017627925 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.113345385 0.17435084 0.018189587 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.069113895 0.012352899 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.28566658 0.34300485 0.02445282 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.05273824 0.46236274 0.14893818 0.007474713 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.019572072 0.02565581 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015046522 0.1898294 0.04390277 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.8118491e-05 0.07695871 0.016719356 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.004143432 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.14339727 0.43230549 0.067207575 0.0058289766 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.056784533 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.039619282 0.17262521 0.027965259 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0912444 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.123520195 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0017318651 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.045247905 0.0467966 0.011087887 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015420213 0.011461236 0.0 0.0 0.0 0.0 \
0.051597454 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.047117434 0.028115034 0.007643938 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.009258673 0.36238867 0.33478934 0.026568115 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.070717014 0.4215075 0.061776347 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.013473757 0.34986925 0.31940612 0.036531948 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.08099596 0.07464287 0.0028904527 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.034357864 0.30965418 0.05155325 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.060108937 0.06775363 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.005902812 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0'

contentlist = content.split(' ')

minnum, maxnum = 210000000, -10000000

arr = []
totalnum = 0
for i in range(33):
    tmparr = []
    for j in range(50):
        number = float(contentlist[i*50+j])
        tmparr.append(number*255)
        totalnum += float(number)

        if minnum > number:
            minnum = number
        if maxnum < number:
            maxnum = number
    arr.append(tmparr)

print(arr)
print(totalnum)
print(minnum, maxnum)

import pylab as plt
import numpy as np

# [1, 3, 528, 800]
xside = np.linspace(0, 50, 50)  # (최소값, 최대값, 격자 수). side = (-2, 2, 4)인 경우, [-2.  -0.66666667  0.66666667  2. ]가 나옴. 즉 차=4, 간=3이니 4/3=1.3333... 차이의 배열이 생김.
yside = np.linspace(0, 33, 33)
X, Y = np.meshgrid(xside,yside)

# Z = np.exp(-((X-1)**2+Y**2))  # 각 격자에 거기에 들어갈 값
arr = np.flip(arr)  # plot은 아래에서부터 0이기 때문에 역순이라 뒤집어줘야됨.
Z = np.array(arr)

# Plot the density map using nearest-neighbor interpolation
plt.pcolormesh(X,Y,Z)
plt.show()

img = np.zeros((528, 800, 3))


### (32, 56)
content2 = '0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09280482 0.097344 0.10841618 0.09139161 0.077246666 0.06828907 0.08848926 0.11435663 0.18256304 0.4149024 0.44499713 0.40994072 0.28752226 0.10366124 0.03777496 0.037533544 0.08418658 0.08656987 0.12326978 0.24217872 0.16407637 0.060826812 0.056939133 0.02794481 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015332513 0.12380709 0.10592283 0.06603281 0.088949256 0.12787008 0.14101663 0.13421753 0.1405271 0.15372358 0.2574297 0.6421821 0.7116873 0.60735714 0.38143605 0.15863875 0.116072044 0.04764746 0.068032324 0.07988216 0.089688346 0.10586059 0.09796429 0.10228443 0.08233502 0.0308013 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07614177 0.110890575 0.13969254 0.21795315 0.25155783 0.17176068 0.13885036 0.11007114 0.13487385 0.21911708 0.46903 0.87972796 0.89708066 0.7606642 0.4825768 0.1627428 0.1123354 0.0614634 0.046394344 0.06539415 0.09617503 0.09250787 0.06859245 0.052645467 0.047115322 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07836355 0.09713887 0.1280506 0.31574702 0.48507664 0.5757699 0.5532752 0.5033391 0.4210778 0.39084136 0.5169118 0.8265995 1.0689806 0.9356024 0.7649851 0.5583403 0.22353485 0.11639148 0.05625426 0.0 0.0 0.026979692 0.046401374 0.04459399 0.0381742 0.028194673 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.041220758 0.13283996 0.1330974 0.46280852 0.7640811 0.83814645 1.0305682 1.3285943 1.4830134 1.3390188 1.1322613 1.1306139 1.264701 1.2226613 0.9554411 0.7943908 0.6867423 0.3636264 0.13257919 0.064790815 0.0 0.0 0.0 0.0 0.0021404326 0.00783816 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.03930672 0.10644765 0.1359361 0.24073167 0.63703954 0.88113713 0.9260926 1.0671521 1.3098816 1.4361649 1.2613469 1.0910344 1.0904976 1.1604568 1.1238881 0.9249852 0.80838156 0.71174574 0.35289064 0.10516809 0.08122346 0.022667386 0.040193185 0.042046867 0.054483373 0.024586044 0.033057943 0.0123067945 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.04742667 0.09251996 0.07338126 0.14609133 0.3929737 0.7056236 0.8748617 0.9351653 1.0347532 1.1913288 1.2408314 1.0280889 0.8757856 0.9139497 1.0058856 1.0377676 0.92136765 0.78224146 0.5809427 0.19093528 0.09733309 0.096214086 0.08649988 0.06434562 0.083508685 0.15590394 0.06446385 0.04922321 0.07646242 0.04860584 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.051978298 0.086605124 0.078758836 0.122785226 0.3920183 0.6031506 0.7462282 0.8261839 0.8570856 0.9082457 0.9934449 0.9894011 0.8194159 0.72955525 0.81112313 0.92088115 0.9487153 0.86612284 0.72742665 0.5044909 0.18966931 0.097058535 0.11388664 0.118672706 0.093145765 0.06598905 0.03971988 0.040477302 0.07627158 0.09857104 0.037664108 0.087318376 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.03192051 0.10157565 0.092459865 0.113534 0.21754833 0.39047065 0.5048436 0.61886775 0.7045795 0.7353002 0.75019276 0.7637963 0.7601439 0.73420703 0.7485169 0.8289876 0.8877218 0.86119103 0.8036406 0.7339356 0.5896244 0.29406136 0.15931848 0.15379632 0.15794289 0.22496289 0.31097952 0.20486535 0.14348888 0.12345263 0.09087584 0.036009293 0.09178563 0.02622084 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.10808227 0.09428926 0.12189782 0.24201861 0.2888356 0.3147265 0.3651207 0.46596026 0.5588813 0.59037006 0.60669196 0.613289 0.63942695 0.70448077 0.78673744 0.8531016 0.8690691 0.8457639 0.8293104 0.8077955 0.7032281 0.4802117 0.29791903 0.23699355 0.24693945 0.34613183 0.40189832 0.2918415 0.20277163 0.15452942 0.09885106 0.053031716 0.086430036 0.05099897 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01819694 0.112942114 0.10842645 0.16149299 0.40137175 0.5014714 0.43144488 0.3864154 0.37006736 0.3945169 0.43502674 0.4569757 0.4788396 0.49665502 0.5121762 0.5360024 0.58554745 0.6375581 0.6768645 0.7183505 0.79729736 0.843297 0.7261958 0.5108175 0.39717394 0.32695997 0.32994768 0.38445807 0.37029353 0.28237498 0.20796552 0.15004025 0.103648946 0.070153445 0.099001214 0.05413611 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.03193369 0.11111929 0.1016291 0.16343564 0.35143086 0.488494 0.5383487 0.5155822 0.4838428 0.43186918 0.39389172 0.3743381 0.36945558 0.37792176 0.38446522 0.37248623 0.36027515 0.38194245 0.41866535 0.46340117 0.5233811 0.6193032 0.7159544 0.6442499 0.49891022 0.41909936 0.36638957 0.36120537 0.37943482 0.33537146 0.26512313 0.19002214 0.12867944 0.10101533 0.084696144 0.08634825 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.018055722 0.10444647 0.098379865 0.18344645 0.36553663 0.4262689 0.41691956 0.4579442 0.49004775 0.49806178 0.45203438 0.37569603 0.31521085 0.29324543 0.28060943 0.27979797 0.29480577 0.31839287 0.34573326 0.3664317 0.39258164 0.43810177 0.49989486 0.5342215 0.50782466 0.477881 0.44107807 0.38426802 0.34399095 0.32540202 0.2890872 0.24866287 0.18312353 0.1133437 0.09860878 0.07700449 0.044668972 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.0 0.0033285767 0.102975935 0.112236515 0.13460575 0.31515607 0.4829944 0.45658883 0.3955373 0.4104367 0.4317596 0.45535097 0.4361842 0.35821974 0.2885766 0.25512046 0.23724817 0.23931427 0.27074122 0.3333323 0.38235483 0.36934614 0.37678283 0.43648723 0.49593955 0.4608993 0.41415146 0.42955416 0.42262515 0.35257956 0.28420678 0.25598612 0.2442933 0.22868384 0.17803797 0.0978048 0.088733144 0.07766072 0.017503113 0.0 0.0 0.0 0.0 0.0 0.0 0.04372726 0.057305895 0.008185513 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.0 0.04749028 0.1169557 0.1261109 0.122919306 0.20833167 0.37478745 0.5007329 0.45930856 0.37168503 0.36810532 0.39326224 0.4388669 0.45025918 0.39206788 0.30054465 0.2455044 0.22536223 0.23523822 0.2707274 0.3797971 0.41745827 0.3412763 0.3296226 0.42678395 0.5331192 0.47107545 0.3908252 0.3813628 0.36393642 0.30137128 0.24053246 0.2254059 0.22728597 0.20725077 0.16221413 0.103800006 0.09123209 0.08105062 0.044954557 0.0 0.0 0.0 0.0 0.0 0.031138651 0.095047 0.105193876 0.08104204 0.0 0.0 0.0 \
0.0 0.0 0.0 0.0 0.07527253 0.11251105 0.10698895 0.12957935 0.2168899 0.33531266 0.41252497 0.4369227 0.37749466 0.3135993 0.30633026 0.33713144 0.41491815 0.46717706 0.44489884 0.34853673 0.27289546 0.24309039 0.25052005 0.3014569 0.41572356 0.39914426 0.3020633 0.2898787 0.36831257 0.50237393 0.45740834 0.3734801 0.35250914 0.33448482 0.29105926 0.25022918 0.24997969 0.2514677 0.21774131 0.17055167 0.12816472 0.10436417 0.06403442 0.05683876 0.041350488 0.0 0.0 0.0 0.0 0.0073174685 0.045363437 0.0576536 0.07324383 0.0 0.0 0.0 \
0.0 0.0 0.0 0.048100684 0.07972757 0.07474577 0.19877923 0.29054567 0.41303065 0.428334 0.40891334 0.3715234 0.3077067 0.25487673 0.24802771 0.2627388 0.34025967 0.4349132 0.44285104 0.36839935 0.3142546 0.28667742 0.2929135 0.34180418 0.4151521 0.38080135 0.29582214 0.27199703 0.31434336 0.39711428 0.36826226 0.3181464 0.31083998 0.31489992 0.30376637 0.2888324 0.30660358 0.30207786 0.25657374 0.20045847 0.15529639 0.113085315 0.0655813 0.10269388 0.058409892 0.041818082 0.0 0.0 0.0 0.0 0.04296304 0.051971417 0.052383643 0.0 0.0 0.0 \
0.0 0.0 0.0056989416 0.07934713 0.05646711 0.1971902 0.4623074 0.41368365 0.40608487 0.3724747 0.36167353 0.34964633 0.2970672 0.22826286 0.2158938 0.22383158 0.26991194 0.3417969 0.35257494 0.3173474 0.31056255 0.3232392 0.34861296 0.39456564 0.43149558 0.3979706 0.30974802 0.25972384 0.2627051 0.2827898 0.2425001 0.2236147 0.25016326 0.29417473 0.32170528 0.3040899 0.28966585 0.2783603 0.25070658 0.19926809 0.15410386 0.12291947 0.08041274 0.06140717 0.057549667 0.08078538 0.07541382 0.036260553 0.024753742 0.021517538 0.04747437 0.055798348 0.042552702 0.014627151 0.0 0.0 \
0.0 0.0 0.05750699 0.09298988 0.083948344 0.30547822 0.4991628 0.3653459 0.31368405 0.2839656 0.29048654 0.35279384 0.3402137 0.23946257 0.20201865 0.21426418 0.2409748 0.27694416 0.2725464 0.26166767 0.28403503 0.34968123 0.41949907 0.4595136 0.4760629 0.43007213 0.33135563 0.26179582 0.24395263 0.2188744 0.18447375 0.18825369 0.21672966 0.28059393 0.33042964 0.2844538 0.2288741 0.19984849 0.18643263 0.16059637 0.12938507 0.10996002 0.079555176 0.053998437 0.05284245 0.07639977 0.0736534 0.07997781 0.066494316 0.0337161 0.017043471 0.035290018 0.054090254 0.04480154 0.02355019 0.015645348 \
0.0 0.042194195 0.07012992 0.07700537 0.12550214 0.31499925 0.40984064 0.32123226 0.26889154 0.2374685 0.25198656 0.34084395 0.3868551 0.28253067 0.22676519 0.22821634 0.24601233 0.25542107 0.24411005 0.2443859 0.27716184 0.36173996 0.44684753 0.46479878 0.4661313 0.41080898 0.32025927 0.26168525 0.2391445 0.21243882 0.18450704 0.18831286 0.21585879 0.2598772 0.28206122 0.23224413 0.17923331 0.16041061 0.15219209 0.12923118 0.106237665 0.104872435 0.10622006 0.09563348 0.060378432 0.051580656 0.07304096 0.09878214 0.06603836 0.0 0.0 0.0 0.01302667 0.05132047 0.018805072 0.014428318 \
0.0 0.0655248 0.044930365 0.054593973 0.17084439 0.30925953 0.33890155 0.31025076 0.27797174 0.24158894 0.2325991 0.27299243 0.31453818 0.28577822 0.26664877 0.28032884 0.28228635 0.26060826 0.2409127 0.24507819 0.279409 0.3445122 0.36449793 0.32740623 0.29911196 0.26172364 0.22645096 0.20825446 0.20928994 0.22798614 0.2268537 0.21833895 0.2227545 0.23291254 0.22076519 0.1853038 0.16372338 0.15107708 0.13900709 0.1081828 0.08014013 0.08969819 0.11124197 0.107548974 0.06821116 0.06298697 0.083932854 0.074800745 0.012170814 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.009754188 0.087766886 0.051550016 0.081035756 0.22718722 0.3150934 0.30573207 0.31054753 0.31192333 0.26663238 0.23090158 0.23055221 0.2403759 0.24969819 0.27221385 0.30873123 0.30315 0.27031547 0.24528521 0.24800897 0.27537096 0.29676273 0.24773481 0.18761012 0.13974304 0.106276855 0.11731765 0.14820418 0.16757831 0.24317287 0.29743177 0.2686262 0.2382935 0.22006412 0.198675 0.17962876 0.1571644 0.15019044 0.1446994 0.12473828 0.089114495 0.10215095 0.124168694 0.114879936 0.080416024 0.06983853 0.07212049 0.04361113 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.05498554 0.11432445 0.09274166 0.13319184 0.26630998 0.28704536 0.24788675 0.26082864 0.2701224 0.2320417 0.21332246 0.21044847 0.2114779 0.23666142 0.25936034 0.29622638 0.3073265 0.2742734 0.23720828 0.22543378 0.2356624 0.22592258 0.18896872 0.16371572 0.1302504 0.10481188 0.12262633 0.14863142 0.16189888 0.2270613 0.28298646 0.26743746 0.23442079 0.21768282 0.20189866 0.18240494 0.16169286 0.15994474 0.17192264 0.17630763 0.17573395 0.19533929 0.20346412 0.17690678 0.09525229 0.09179013 0.079843216 0.054421756 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 \
0.10748199 0.12907597 0.13383755 0.2120826 0.30282745 0.25039428 0.19829226 0.19935931 0.18253842 0.15998347 0.15786141 0.15701327 0.13749518 0.18074544 0.2171786 0.26286954 0.31393358 0.27259737 0.1984213 0.16111416 0.15071064 0.11551075 0.11398366 0.1146815 0.12398182 0.110481195 0.121181995 0.115695365 0.1303435 0.21023971 0.2393715 0.2335045 0.22982207 0.2290953 0.20959294 0.18856236 0.18401603 0.19310796 0.21001822 0.22800347 0.25667062 0.30906063 0.3243729 0.24879923 0.10713293 0.08863968 0.08940743 0.08778207 0.07539365 0.056924324 0.0 0.0 0.0 0.0 0.0 0.0 \
0.20501617 0.19000667 0.22055992 0.28384238 0.2849623 0.22018537 0.19975567 0.1904541 0.1646561 0.14591089 0.14577633 0.1377825 0.1131447 0.14199272 0.168125 0.19353847 0.2074636 0.18570182 0.1594931 0.16047452 0.1454848 0.13959312 0.15902947 0.10201803 0.1337229 0.11590028 0.09182544 0.08506416 0.087634645 0.15091614 0.22323185 0.2370883 0.26604015 0.26860327 0.23336807 0.20632346 0.20016281 0.2102842 0.2190783 0.21231887 0.24769108 0.31618217 0.33706123 0.2896564 0.13438141 0.094828255 0.08659841 0.095876716 0.07874437 0.06156915 0.06424149 0.02236794 0.0 0.0 0.0 0.0 \
0.29520923 0.28421447 0.27930483 0.26156944 0.22161607 0.19526297 0.19716015 0.19565631 0.1700865 0.14180723 0.15380096 0.14964706 0.1386098 0.15204501 0.16554084 0.13969737 0.110474855 0.10426562 0.12105645 0.1255232 0.09708497 0.09882834 0.12919882 0.11583201 0.16286817 0.115693465 0.08572696 0.14361867 0.15439525 0.12229514 0.20870465 0.22392797 0.26746064 0.28465366 0.24778426 0.21748617 0.2090169 0.21089968 0.21205431 0.1813586 0.21572165 0.25682953 0.27633 0.23927176 0.16198444 0.12577045 0.12802656 0.09664306 0.05914458 0.05818897 0.060859554 0.063524626 0.06638619 0.019817375 0.0 0.0 \
0.24300762 0.23822473 0.24630482 0.22515869 0.20029129 0.19418165 0.19910616 0.20908973 0.19211423 0.15772694 0.15976337 0.15612185 0.1584059 0.16399252 0.14791232 0.12614462 0.10692119 0.1091999 0.114104554 0.08022238 0.06385791 0.06978937 0.14362216 0.1301766 0.099027194 0.115208074 0.11532831 0.109228484 0.107654154 0.12535107 0.22151943 0.22131388 0.22955567 0.22353807 0.17832494 0.17362545 0.20734195 0.16689275 0.13066046 0.15791652 0.18449359 0.16370654 0.171067 0.1935611 0.18207414 0.16544689 0.16187505 0.11409839 0.08617731 0.044908382 0.06229262 0.05750218 0.089185104 0.07991401 0.031086363 0.056927066 \
0.18904604 0.15824756 0.23068984 0.20546035 0.1865106 0.19154254 0.19353873 0.2026751 0.20568049 0.18193403 0.16253532 0.15948103 0.18974686 0.2142764 0.1606156 0.12702072 0.13291804 0.12634084 0.12111379 0.08963369 0.2004284 0.09994536 0.094484724 0.13055432 0.09770594 0.120760515 0.13736598 0.10399651 0.06722509 0.09240471 0.12119347 0.14931434 0.15659845 0.16453844 0.14514256 0.1334675 0.13977343 0.10067655 0.118616484 0.10792939 0.12609608 0.14096539 0.16374534 0.1510897 0.105064444 0.103444025 0.1265776 0.115101755 0.10095539 0.057102215 0.069065794 0.059657797 0.05542625 0.02645221 0.047427963 0.04271637 \
0.12619564 0.1066766 0.18279292 0.19238582 0.1974254 0.18926227 0.17203933 0.18313298 0.2356257 0.23658572 0.16842876 0.17518541 0.23736754 0.32571155 0.23669888 0.15428805 0.1505859 0.13751778 0.13385567 0.09478699 0.102646716 0.086545505 0.1286443 0.15038213 0.13455155 0.16277179 0.19087216 0.10630599 0.20998853 0.1857901 0.08388354 0.12967157 0.13138315 0.1256618 0.14200489 0.12830247 0.117922485 0.112532154 0.2044469 0.11886749 0.12932287 0.15113649 0.15881538 0.13517012 0.20029321 0.16479126 0.0726194 0.03652726 0.044187415 0.07147053 0.07021673 0.061104544 0.053466734 0.037809458 0.2167952 0.0887161 \
0.10028174 0.10842857 0.15383577 0.16875367 0.19195792 0.17901917 0.15346153 0.16083078 0.24359049 0.2695188 0.19089434 0.17770135 0.18091188 0.19272903 0.18217865 0.16156814 0.14515099 0.12894882 0.15414035 0.15873465 0.10741171 0.123949796 0.1695335 0.16043629 0.17499778 0.22277473 0.21634029 0.10228965 0.09197629 0.076690674 0.0822107 0.17825405 0.14772482 0.13186625 0.14533067 0.13283575 0.12238255 0.082644545 0.08181285 0.066358484 0.08867595 0.13338217 0.13211255 0.1049921 0.1049079 0.098451704 0.01725483 0.08876072 0.030219298 0.039002504 0.06270804 0.05497622 0.06898268 0.04114545 0.050595105 0.060810946 \
0.08942823 0.10237737 0.11407773 0.11658211 0.1283697 0.13387437 0.14613599 0.16122285 0.1715138 0.16332656 0.13312548 0.13324131 0.10761861 0.09936063 0.09305639 0.14230686 0.13359724 0.121050335 0.15821823 0.17685308 0.12573172 0.124118626 0.14450468 0.16422915 0.17399463 0.10520662 0.05888063 0.06623052 0.06906402 0.06427097 0.0829104 0.110303394 0.123598516 0.17422073 0.164102 0.12583992 0.123830974 0.12906355 0.08704025 0.082165845 0.10233585 0.086279385 0.0869135 0.06928414 0.08756061 0.07963997 0.053349 0.39669383 0.096831575 0.035750642 0.06767696 0.023826525 0.058694586 0.088663146 0.046884 0.05305823 \
0.04845194 0.09161336 0.097781196 0.115018025 0.13200703 0.1282374 0.13773814 0.14883421 0.15384392 0.15563837 0.14753266 0.13767575 0.15801474 0.19016701 0.1363441 0.14786519 0.14654338 0.14697312 0.15021962 0.14026673 0.18625411 0.17798501 0.15469775 0.17226323 0.12665924 0.11467311 0.24183887 0.21264657 0.07742591 0.074470095 0.099446155 0.110691465 0.1451964 0.18680745 0.17978027 0.15838273 0.14950824 0.10830814 0.084466346 0.17478518 0.16485241 0.057835247 0.13274683 0.21854448 0.13237622 0.09978493 0.034048308 0.053005293 0.024138555 0.044104263 0.053584423 0.022329845 0.02079679 0.03360769 0.028635692 0.08100311'
###
