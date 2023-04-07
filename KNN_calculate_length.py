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
