import numpy as np
import py360convert
from PIL import Image
import os
rootpath = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/1796/sphere'
rootpath2 = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/2098/sphere'
rootpath3 = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/2101/sphere'
d_path_direc = []

flag = True
# os.walk() : 인자로 전달된 경로의 폴더를 재귀적으로 탐색, 모든 파일리스트를 출력
for (root, directories, files) in os.walk(rootpath):
    for d in directories: 
        d_path = os.path.join(root, d)
        if d_path.endswith('_E'): # 디렉토리들 중 _E로 끝나는 것들은
            #print(d_path[79:85])
            if d_path[79:85] == '196295' or d_path[79:85] == '196306' or \
                d_path[79:85] == '196311' or d_path[79:85] == '196321' :
                continue

            print('현 디렉토리는 ' + d_path + '내의 ' + d_path[79:-3])
            # d_path_direc.append(d_path)

            filelist = os.listdir(d_path) # 그것들의 파일리스트를 받아와서
            cnt = 0
            for file in filelist:
                file_path = d_path + '/' + file
                original_filename = file[:-4]
                if original_filename.startswith('Thu'):
                    print(file_path)
                    continue

                cnt += 1
                print(file_path)

                e_img = np.array(Image.open(file_path))

                cubemap = py360convert.e2p(e_img, fov_deg = (90, 90), u_deg = -0,
                                           v_deg = 0, out_hw = (2000, 2000), in_rot_deg = 0)
                c_img = Image.fromarray(cubemap)
                cubemap2 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=-90,
                                           v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
                c_img2 = Image.fromarray(cubemap2)
                cubemap3 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=90,
                                           v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
                c_img3 = Image.fromarray(cubemap3)
                cubemap4 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=180,
                                           v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
                c_img4 = Image.fromarray(cubemap4)

                curdirec = d_path[79:-3]
                makedir = 'C:/Users/jsk/PycharmProjects/pythonProject1/1796' + '/' + curdirec
                # makedir = 'C:/Users/jsk/PycharmProjects/pythonProject1/2098' + '/' + curdirec
                # makedir = 'C:/Users/jsk/PycharmProjects/pythonProject1/2101' + '/' + curdirec
                try:
                    if not os.path.exists(makedir):
                        os.makedirs(makedir)
                except OSError:
                    pass

                c_img.save(makedir + '/' + original_filename + 'F' + '.jpg')
                c_img2.save(makedir + '/' + original_filename + 'L' + '.jpg')
                c_img3.save(makedir + '/' + original_filename + 'R' + '.jpg')
                c_img4.save(makedir + '/' + original_filename + 'B' + '.jpg')
                
                
               
#2중 for문 버전

import numpy as np
import py360convert
from PIL import Image
import os


rootpath = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/1796/sphere'
rootpath2 = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/2098/sphere'
rootpath3 = '//mldisk.sogang.ac.kr/nfs_shared_/STR_Data/RoadView/muhan_roadview/2101/sphere'
rootpath = '/nfs_shared/STR_Data/RoadView/muhan_roadview/2098/sphere'
rootpath = '/nfs_shared/STR_Data/RoadView/muhan_roadview/1796/sphere'
d_path_direc = []

flag = True
# os.walk() : 인자로 전달된 경로의 폴더를 재귀적으로 탐색, 모든 파일리스트를 출력
for (root, directories, files) in os.walk(rootpath):

    if (len(directories)) > 0:
        for dir_name in directories:
            if dir_name.endswith('_E'):
                # print("dir: " + root + '/' + dir_name)
                d_path_direc.append(root + '/' + dir_name)

# d_path_direc.pop(0)
d_path_direc.reverse()
# print(d_path_direc)

for path in d_path_direc:
    files = os.listdir(path)
    makedir = '/nfs_shared/STR_Data/RoadView/muhan_roadview_transformed/1796' + '/' + path[57:-3]
    try:
        if not os.path.exists(makedir):
            os.makedirs(makedir)
    except OSError:
        pass

    for file in files:
        # print(path + '/' + file)
        file_path = path + '/' + file
        original_filename = file[:-4]
        if original_filename.startswith('Thu'):
            print(file_path + '/' + original_filename)
            continue

        # print(original_filename)
        # print(makedir + '/' + original_filename + 'F' + '.jpg')

        e_img = np.array(Image.open(file_path))
        cubemap = py360convert.e2p(e_img, fov_deg = (90, 90), u_deg = -0,
                                                v_deg = 0, out_hw = (2000, 2000), in_rot_deg = 0)
        c_img = Image.fromarray(cubemap)
        cubemap2 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=-90,
                                               v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
        c_img2 = Image.fromarray(cubemap2)
        cubemap3 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=90,
                                               v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
        c_img3 = Image.fromarray(cubemap3)
        cubemap4 = py360convert.e2p(e_img, fov_deg=(90, 90), u_deg=180,
                                               v_deg=0, out_hw=(2000, 2000), in_rot_deg=0)
        c_img4 = Image.fromarray(cubemap4)

        c_img.save(makedir + '/' + original_filename + 'F' + '.jpg')
        c_img2.save(makedir + '/' + original_filename + 'L' + '.jpg')
        c_img3.save(makedir + '/' + original_filename + 'R' + '.jpg')
        c_img4.save(makedir + '/' + original_filename + 'B' + '.jpg')



