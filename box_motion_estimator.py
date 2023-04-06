"""
This function takes in a reference frame and a current frame as input, and divides the reference frame into 
non-overlapping blocks of size block_size. It then searches for the best match in the current frame within a search 
range of search_range pixels in each direction using the SAD metric. The motion vectors for each block are saved 
in a 2D array and returned as output.
"""
# Function to calculate the Sum of Absolute Differences (SAD)
def SAD(block1, block2):
    return np.sum(np.abs(block1 - block2))

# Function to perform block matching
def block_matching(ref_frame, cur_frame, block_size=30, search_range=16):
    height, width = ref_frame.shape[:2]
    print(height, width)
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)

    # Divide reference frame into blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            ref_block = ref_frame[i:i+block_size, j:j+block_size]

            # Set initial motion vector to (0, 0)
            mv = np.array([0, 0])
            position = [0, 0, 0, 0]

            # Search for the best match in the current frame
            min_sad = float('inf')
            for k in range(-search_range, search_range+1):  # -32 ~ 32
                for l in range(-search_range, search_range+1):  # =32 ~ 32
                    # Check if search window is within the frame
                    # block_height_idx + k >= 0 and block_height_idx + k + 16 < height and ...
                    if i+k >= 0 and i+k+block_size < height and j+l >= 0 and j+l+block_size < width:
                        cur_block = cur_frame[i+k:i+k+block_size, j+l:j+l+block_size]
                        sad = SAD(ref_block, cur_block)
                        if sad <= min_sad:
                            min_sad = sad
                            mv = np.array([k, l])
                            position = [j+l, i+k, j+l+block_size, i+k+block_size]

            # Save motion vector for the current block
            motion_vectors[i//block_size, j//block_size] = mv
            if position[0] != i :
                img = cv2.rectangle(cur_frame, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 3)  # find box, green
                # img = cv2.rectangle(cur_frame, (j, i), (j + block_size, i + block_size), (0, 0, 255), 3)  # original box, red
            else :
                img = cv2.rectangle(cur_frame, (position[0], position[1]), (position[2], position[3]), (0, 255, 255), 3)  # find box, green

            if i < 690 :
                cv2.imshow('sample', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



    return motion_vectors

if __name__ == "__main__":
    listdr = os.listdir('./as')
    for i in range(len(listdr)):
        listdr[i] = cv2.resize(cv2.imread(os.path.join('./as', listdr[i])), (960, 720))
    mt_vt = block_matching(listdr[0], listdr[1])
