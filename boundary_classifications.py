import open3d as o3d
import numpy as np
import scipy as sc
import copy
import point_cloud_utils as pcu
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import itertools

from numpy.linalg import norm


#DBSCAN Clustering
SOURCE_PCD =  o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/7.ply")
pcd = SOURCE_PCD.sample_points_poisson_disk(number_of_points = 10000)
pcdd = copy.deepcopy(pcd)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n = 3, num_iterations=10000)
[a, b, c, d] = plane_model


inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
segment_models={}
segments={}
max_plane_idx=10
rest=pcd
pcl = o3d.geometry.PointCloud()
cl = []




for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold = 0.01,ransac_n = 3,num_iterations=10000)
    segments[i] = rest.select_by_index(inliers)
    print("inliers")
    print("points")
    print(segments[i])
    segments[i].paint_uniform_color(list(colors[:3]))
    pcl += segments[i]
    cl.append(np.asarray(segments[i].colors)[0])
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")


pcl += rest


pcn = o3d.geometry.PointCloud()
pcn=rest
o3d.visualization.draw_geometries([pcl])

o3d.visualization.draw_geometries([rest])

pcd_tree = o3d.geometry.KDTreeFlann(pcl)
distances = np.max(pcl.compute_nearest_neighbor_distance())

dic = []

hull, _ = rest.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
box = hull_ls.get_axis_aligned_bounding_box()
v_ = box.volume()
v_  = round(v_,0) 



for pointt in np.asarray(rest.points):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pointt, distances + (v_ /1000000))
    #[k, idx, _] = pcd_tree.search_knn_vector_3d(pointt, 4)
    arr = [0] * (int(max_plane_idx/2)+3)
    for i in np.asarray(pcl.points)[idx[1:], :]:
        plane_idx = 0
        arr_idx = 0
        while plane_idx != (max_plane_idx):
            #check which segments are around the current black point 
            if (i in np.asarray(segments[plane_idx].points)) or  (i in np.asarray(segments[plane_idx+1].points)) :
                print("segment0")
                arr[arr_idx] = 1
            plane_idx += 2
            arr_idx += 1
            
    print("---------------------------------------------")
    #completed the process for one black point
    arr[arr_idx] = pointt[0]
    arr[arr_idx + 1] = pointt[1]
    arr[arr_idx + 2] = pointt[2]
    #add black point neighbour's info to dictionary      
    dic.append(arr)
 

arr_ = []
visited = [False] * len(dic)
n = 0

for i in range(0,len(dic)):
    point_ = []
    if visited[i]== False:
        visited[i] = True
        n = 0
        point_.append([dic[i][int((len(arr)-3))],dic[i][int((len(arr)-2))],dic[i][int((len(arr)-1))]])
    
        for a in range(i+1,len(dic)):
            if visited[a]== False:
                true_all = 0
                #if (abs(dic[i][0]-dic[a][0])<1) & (abs(dic[i][1]-dic[a][1]) < 1)& (abs(dic[i][2]-dic[a][2]) < 1)& (abs(dic[i][3]-dic[a][3]) < 1) & (abs(dic[i][4]-dic[a][4])< 1):
                for dic_ind in range(0,(len(arr)-3)):
                    if (dic[i][dic_ind]==dic[a][dic_ind]):
                        true_all += 1
                
                if true_all == (len(arr)-3):
                    point_.append([dic[a][int((len(arr)-3))],dic[a][int((len(arr)-2))],dic[a][int((len(arr)-1))]])
                    visited[a] = True
                    n = 1
                
        #holds black points have same neighbours
        if n==1:
            arr_.append(point_)

            
            

x = 0
y = 0
rest.paint_uniform_color([1, 1, 1])

for i in arr_: 
    # a is black point
    if len(i)< 20:
        continue
    for a in i:
        rest.points[x] = a
        rest.colors[x] = cl[y]
        x += 1
        
    y +=1
    if y == max_plane_idx:
        y=0
        


o3d.visualization.draw_geometries([rest])

