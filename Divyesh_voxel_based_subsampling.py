

import pyvista as pv
import numpy as np

# Load the 3D point cloud data
cloud = pv.read("point_cloud_data.ply")

# Define the voxel size
voxel_size = 0.1

# Convert the point cloud to a 3D voxel grid
grid = cloud.voxelize(voxel_size)

# Compute the proportion of each class of points in each voxel
num_classes = 5
class_proportions = np.zeros((grid.dimensions[0], grid.dimensions[1], grid.dimensions[2], num_classes))
for i in range(grid.dimensions[0]):
    for j in range(grid.dimensions[1]):
        for k in range(grid.dimensions[2]):
            voxel_points = grid.extract_cells_within_box(
                [i * voxel_size, (i + 1) * voxel_size,
                 j * voxel_size, (j + 1) * voxel_size,
                 k * voxel_size, (k + 1) * voxel_size])
            for class_id in range(num_classes):
                class_proportions[i, j, k, class_id] = np.count_nonzero(voxel_points["class"] == class_id) / len(voxel_points)

# Select points uniformly from each voxel based on the proportion of each class of points
subsampled_points = []
for i in range(grid.dimensions[0]):
    for j in range(grid.dimensions[1]):
        for k in range(grid.dimensions[2]):
            voxel_points = grid.extract_cells_within_box(
                [i * voxel_size, (i + 1) * voxel_size,
                 j * voxel_size, (j + 1) * voxel_size,
                 k * voxel_size, (k + 1) * voxel_size])
            num_points = len(voxel_points)
            class_probs = class_proportions[i, j, k]
            for class_id in range(num_classes):
                class_indices = np.where(voxel_points["class"] == class_id)[0]
                num_class_points = len(class_indices)
                if num_points > 0:
                    num_samples = int(round(num_points * class_probs[class_id]))
                    if num_class_points < num_samples:
                        subsampled_points.extend

