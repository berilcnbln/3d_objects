import open3d as o3d
import numpy as np
import scipy as sc
import copy
import point_cloud_utils as pcu
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import itertools
import open3d.visualization.gui as gui
import open3d.visualization as vis

from numpy.linalg import norm
import matplotlib.pyplot as plt


vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
SOURCE_PCD = o3d.io.read_triangle_mesh("files/3.ply")
# ---------------------------------------------------------------------------------
pcd = SOURCE_PCD.sample_points_poisson_disk(number_of_points=10000)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
                     fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)
[a, b, c, d] = plane_model

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label
                                         if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
segment_models = {}
segments = {}
max_plane_idx = 10
rest = pcd
pcl = o3d.geometry.PointCloud()
cl = []
i_ = 0

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=10000)
    segments[i] = rest.select_by_index(inliers)
    print("inliers")
    print("points")
    print(segments[i])
    # print(np.asarray(pcd.select_by_index(inliers).points))
    segments[i].paint_uniform_color(list(colors[:3]))
    pcl += segments[i]
    cl.append(np.asarray(segments[i].colors)[0])
    rest = rest.select_by_index(inliers, invert=True)
    print("pass", i, "/", max_plane_idx, "done.")

pcl += rest


# o3d.visualization.draw_geometries([pcl])
# o3d.visualization.draw_geometries([rest])


pcn = o3d.geometry.PointCloud()
pcn = rest

pcd_tree = o3d.geometry.KDTreeFlann(pcl)
distances = np.max(pcl.compute_nearest_neighbor_distance())

blacks = []

# holds data of each black points' neighbours
dic = []

for pointt in np.asarray(rest.points):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pointt, distances)
    # [k, idx, _] = pcd_tree.search_knn_vector_3d(pointt, 4)
    arr = [0] * (int(max_plane_idx / 2) + 3)
    n = 0
    for i in np.asarray(pcl.points)[idx[1:], :]:
        plane_idx = 0
        arr_idx = 0
        while plane_idx != (max_plane_idx):
            # check which segments are around the current black point
            if (i in np.asarray(segments[plane_idx].points)) or (i in np.asarray(segments[plane_idx + 1].points)):
                print("segment0")
                arr[arr_idx] = 1
                n = 1
            plane_idx += 2
            arr_idx += 1

    print("---------------------------------------------")
    # completed the process for one black point
    print(arr_idx)
    if n != 0:
        arr[arr_idx] = pointt[0]
        arr[arr_idx + 1] = pointt[1]
        arr[arr_idx + 2] = pointt[2]
        # add processed black point neighbour's info to dic
        dic.append(arr)

arr_ = []
visited = [False] * len(dic)
n = 0

print("dic lenght" + str(len(dic)))


for i in range(0, len(dic)):
    point_ = []
    if visited[i] == False:
        visited[i] = True
        n = 0
        point_.append([dic[i][int((len(arr) - 3))], dic[i][int((len(arr) - 2))], dic[i][int((len(arr) - 1))]])

        for a in range(i + 1, len(dic)):
            if visited[a] == False:
                true_all = 0
                # if (abs(dic[i][0]-dic[a][0])<1) & (abs(dic[i][1]-dic[a][1]) < 1)& (abs(dic[i][2]-dic[a][2]) < 1)& (abs(dic[i][3]-dic[a][3]) < 1) & (abs(dic[i][4]-dic[a][4])< 1):
                for dic_ind in range(0, (len(arr) - 3)):
                    if (dic[i][dic_ind] == dic[a][dic_ind]):
                        true_all += 1

                if true_all == (len(arr) - 3):
                    point_.append(
                        [dic[a][int((len(arr) - 3))], dic[a][int((len(arr) - 2))], dic[a][int((len(arr) - 1))]])
                    visited[a] = True
                    n = 1

        # holds black points have same neighbours
        if n == 1:
            arr_.append(point_)

x = 0
y = 0
c_index = []

rest.paint_uniform_color([1, 1, 1])
for i in arr_:
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np.random.randn(len(i), 3))
    b = 0
    if len(i) < 30:
        pass
    c_i = []
    for a in i:
        rest.points[x] = a
        pcl.points[b] = a
        b += 1
        c_i.append(x)
        rest.colors[x] = cl[y]
        x += 1
    # print(cll[y])

    # color different color
    c_index.append(c_i)
    y += 1
    if y == max_plane_idx:
        y = 0



vis.add_geometry(rest)
vis.run()
vis.destroy_window()
print(vis.get_picked_points())


for i in vis.get_picked_points():
    for a in range(0, len(c_index)):
        if (i in np.asarray(c_index[a])):
            print("point:  " + str(rest.points[i]) + "  in cluster  " + str(a))

