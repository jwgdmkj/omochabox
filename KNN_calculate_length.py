import numpy as np
from sklearn.neighbors import NearestNeighbors

# Previous frame bounding boxes (x, y, w, h)
prev_boxes = np.array([[100, 200, 50, 80], [150, 180, 60, 90]])

# Next frame bounding boxes (x, y, w, h)
next_boxes = np.array([[110, 190, 50, 80], [140, 185, 60, 90]])

# Initialize KNN model with k=1
knn = NearestNeighbors(n_neighbors=1)

# Fit the KNN model with the previous frame bounding boxes
knn.fit(prev_boxes)

# Find the nearest neighbor for each next frame bounding box
distances, indices = knn.kneighbors(next_boxes)

# Compute the displacement between matched boxes
displacement = next_boxes - prev_boxes[indices[:, 0]]

# The displacement array now contains the flow of the crowd between the two frames
print(displacement)

# --------------------------------------------# 

import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

# Function to create feature vectors from the image and bounding boxes
# output : resize the bounding box into 32x32, and flatten it.
def create_feature_vectors(image, boxes):
    feature_vectors = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Crop the image using bounding box coordinates
        crop = image[y1:y2, x1:x2]
        # Resize the crop to a fixed size
        crop = cv2.resize(crop, (32, 32))
        # Convert the crop to a feature vector
        feature_vector = crop.flatten() # crop : [32, 32, 3]
        feature_vectors.append(feature_vector)
    return np.array(feature_vectors)

# Load the images and bounding boxes
prev_frame = cv2.imread('./data/00000.jpg')
prev_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
next_frame = cv2.imread('./data/00001.jpg')
next_boxes = np.array([[120, 120, 220, 220], [280, 280, 380, 380]])

# Create feature vectors for the previous and next frames
prev_features = create_feature_vectors(prev_frame, prev_boxes)
next_features = create_feature_vectors(next_frame, next_boxes)


# Use KNN to find the nearest neighbor for each bounding box in the previous frame
k = 1 # number of neighbors to find
knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(next_features)
distances, indices = knn.kneighbors(prev_features)
print(distances)
print(indices)

# Calculate the flow vector for each bounding box in the previous frame
flow_vectors = []
for i, box in enumerate(prev_boxes):
    x1, y1, x2, y2 = box
    dx = next_boxes[indices[i][0]][0] - x1
    dy = next_boxes[indices[i][0]][1] - y1
    flow_vectors.append([dx, dy])

# Display the flow vectors on the image
for i, box in enumerate(prev_boxes):
    x1, y1, x2, y2 = box
    cv2.rectangle(prev_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    dx, dy = flow_vectors[i]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.arrowedLine(prev_frame, (cx, cy), (cx+dx, cy+dy), (0, 0, 255), 2)

# Display the image
cv2.imshow("Flow estimation", prev_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
