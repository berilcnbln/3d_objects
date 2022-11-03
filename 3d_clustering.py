#!pip install plotly.express
import pandas as pd
import open3d as o3d
import numpy as np
import copy
import point_cloud_utils as pcu
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


#read mesh files
SOURCE_PCD =  o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/3.ply")
print("start")
pcds = o3d.io.read_point_cloud(r"/Users/beril/PycharmProjects/pythonProject2/files/0.ply")

pcd = SOURCE_PCD.sample_points_poisson_disk(number_of_points=10000)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n = 3, num_iterations=10000)
[a, b, c, d] = plane_model

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])
segment_models={}
segments={}
max_plane_idx=20
rest=pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold = 0.01,ransac_n = 3,num_iterations=10000)
    
    
    segments[i] = rest.select_by_index(inliers)
    print("points")
    print(segments[i])
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")


o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])



#normal vector calculations
distances = np.max(rest.compute_nearest_neighbor_distance())
nb_neighbors = 5
o3d.visualization.draw_geometries([rest])

rest.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))  # invalidate existing normals

rest.estimate_normals()
rest.orient_normals_consistent_tangent_plane(100)
rest.normalize_normals()


