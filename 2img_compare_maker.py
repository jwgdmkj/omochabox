import cv2
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def save(line,save_to):
    line = ' '.join([str(i) for i in line]) + '\n'
    with open(save_to, "a") as f:
        f.write(line)


def draw(paths, sup_title, title):
    start = time.time()
    fig = plt.figure(figsize=(18, 5), clear=True)
    axes = [fig.add_subplot(1, 2, 1),
            fig.add_subplot(1, 2, 2),
            # fig.add_subplot(1, 3, 3),
            # fig.add_subplot(1, 5, 4),
            # fig.add_subplot(1, 5, 5),
            # fig.add_subplot(1, 6, 6),
            ]
    r = [[ax.set_xticks([]),
          ax.set_yticks([])] for ax in axes]

    fig.suptitle(sup_title, fontsize=20)
    axes[0].set_title(title + '\n' + os.path.basename(paths[0]), fontsize=10)
    [axes[n].set_title(os.path.basename(path), fontsize=15) for n, path in enumerate(paths[1:], start=1)]

    # [axes[n].set_xlabel(label, fontsize=10) for n, label in enumerate(xlabels)]
    # [axes[n].set_ylabel(label, fontsize=8) for n, label in enumerate(ylabels)]
    [axes[n].imshow(plt.imread(path)) for n, path in enumerate(paths)]

    end = time.time()

    print(f"{end - start:.5f} sec")

    # axes[0].imshow(images[0]),
    fig.canvas.draw()

    end = time.time()

    print(f"{end - start:.5f} sec")

    im = cv2.cvtColor(np.array(fig.canvas.renderer._renderer), cv2.COLOR_BGR2RGB)

    plt.close(fig)
    end = time.time()

    print(f"{end - start:.5f} sec")
    return im


if __name__ == '__main__':

    # 아래의 것들을 다 합친 절대경로 구성 ex. \\\\mldist.sogang.ac.kr\nfs_shared\...\RoadView\muhan...
    base = os.path.join('\\\\mldisk.sogang.ac.kr', 'nfs_shared_', 'STR_Data', 'RoadView', 'muhan_roadview_resized')
    # base = os.path.join('\\\\mldisk.sogang.ac.kr', 'nfs_shared_', 'STR_Data', 'RoadView', 'muhan_roadview_transformed')
    # base = 'E:'
    # save_to='RoadView_VR_matching.txt'
    save_to = 'RoadView_REAL_matching.txt'
    # cand_file = open("matching_list_2098.txt")
    cand_file = open("candidate.txt")

    lines = cand_file.readlines()  # cand_file 내 모든 줄을 읽어, 각 줄을 리스트로 return

    start = 1111

    total = len(lines)

    # bar = tqdm(lines)
    # cv2.namedWindow("select", cv2.WINDOW_KEEPRATIO)
    # values = sorted(df.values, key=lambda x: x[11], reverse=True)
    values = lines[start:]  # values : 읽어들인 line을, start부터 리스트로 return

    for i, l in enumerate(values,start=start):
        query_path, db_paths = l.split()[0], l.split()[1:]
        print(lines[i].split()[0], l.split()[0])

        select_db_path = db_paths[0]

        query_path_split = query_path.split('/')

        query_1 = query_path_split[-1]
        query_2 = query_path_split[-2]
        query_3 = query_path_split[-3]
        query_4 = query_path_split[-4]
        query_5 = query_path_split[-5]

        # paths = [os.path.join(base, query_1, query_2, query_3)] + [os.path.join(base, b, b_frames[n]) for n, bi in
        #                                             enumerate(b_frame_idx) if bi != -1]

        # db_path_chunks = list_chunk(db_paths, 2)

        sup_title = f'{i} / {total}'

        query_path = os.path.join(base, query_5, query_4, query_3, query_2, query_1)
        db_idx = 0
        db_total = len(db_paths)
        while db_idx < db_total:
            db_path = db_paths[db_idx]
            paths = [query_path] + [
                os.path.join(base, db_path.split('/')[-5], db_path.split('/')[-4], db_path.split('/')[-3],
                             db_path.split('/')[-2], db_path.split('/')[-1])]
            title=f'db_index: {db_idx+1} / {db_total}\n'
            images = draw(paths, sup_title, title)

            cv2.imshow("select", images)
            key = cv2.waitKey()
            while not (ord('0') <= key <= ord('1') or key == ord('b') or key == ord('q') or key == ord('n') or key == ord('x')):
                key = cv2.waitKey()

            if key == ord('1'):
                select_db_path = os.path.join(db_path.split('/')[-5], db_path.split('/')[-4], db_path.split('/')[-3],
                             db_path.split('/')[-2], db_path.split('/')[-1])
                print(query_path)
                select_query_path = os.path.join(query_5, query_4, query_3, query_2, query_1)
                save([select_query_path, select_db_path],save_to)
                break
            if key == ord('n'):
                if db_idx + 1 < db_total:
                    db_idx += 1
                continue
            if key == ord('b'):
                if db_idx > 0:
                    db_idx -= 1
                continue
            if key == ord('x'):
                select_db_path = 'None'
                save([os.path.basename(query_path), os.path.basename(select_db_path)], save_to)
                break
            if key == ord('q'):
                break
                # db_total = len(db_path_chunks)
        # for n, db_path_chunk in enumerate(db_path_chunks):
        #     paths = [query_path] + [os.path.join(base, db_path.split('/')[-5], db_path.split('/')[-4],db_path.split('/')[-3], db_path.split('/')[-2], db_path.split('/')[-1] ) for n, db_path in enumerate(db_path_chunk)]
        #     title=f'db_index: {n} / {db_total}\n'
        #     images = draw(paths, sup_title, title)
        #
        #     cv2.imshow("select", images)
        #     key = cv2.waitKey()
        #     while not (ord('0') <= key <= ord('4') or key == ord('x') or key == ord('q') or key == ord('n')):
        #         key = cv2.waitKey()
        #
        #     if key == ord('1'):
        #         select_db_path = db_path_chunk[0]
        #         save([os.path.basename(query_path), os.path.basename(select_db_path)],save_to)
        #         break
        #     elif key == ord('2'):
        #         select_db_path = db_path_chunk[1]
        #         save([os.path.basename(query_path), os.path.basename(select_db_path)],save_to)
        #         break
        #     elif key == ord('3'):
        #         select_db_path = db_path_chunk[2]
        #         save([os.path.basename(query_path), os.path.basename(select_db_path)],save_to)
        #         break
        #     elif key == ord('4'):
        #         select_db_path = db_path_chunk[3]
        #         save([os.path.basename(query_path), os.path.basename(select_db_path)],save_to)
        #         break
        #     elif key == ord('5'):
        #         select_db_path = db_path_chunk[4]
        #         save([os.path.basename(query_path), os.path.basename(select_db_path)],save_to)
        #         break
        #     elif key == ord('n'):
        #         continue
        if key == ord('q'):
            break
        bar.set_description_str(
            f'{i}/ {total}')
        # cv2.destroyAllWindows()



        bar.update()
