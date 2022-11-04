import open3d as o3d
import numpy as np
import scipy as sc
import copy
import point_cloud_utils as pcu
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


SOURCE_PCD =  o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/3.ply")
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
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#o3d.visualization.draw_geometries([pcd])
segment_models={}
segments={}
max_plane_idx=10
rest=pcd
pcl = o3d.geometry.PointCloud()
cl = []
i_=0

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold = 0.01,ransac_n = 3,num_iterations=10000)
    segments[i] = rest.select_by_index(inliers)
    print("inliers")
    print("points")
    print(segments[i])
    #print(np.asarray(pcd.select_by_index(inliers).points))
    segments[i].paint_uniform_color(list(colors[:3]))
    pcl += segments[i]
    cl.append(np.asarray(segments[i].colors)[0])
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")


pcl += rest

for i in cl:
    print(i)

o3d.visualization.draw_geometries([pcl])
o3d.visualization.draw_geometries([rest])


pcd_tree = o3d.geometry.KDTreeFlann(pcl)
distances = np.max(pcl.compute_nearest_neighbor_distance())


#holds data of each black points neighbours
dic = []
#check all black points until stack is empty
for pointt in np.asarray(rest.points):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pointt, distances)
    #[k, idx, _] = pcd_tree.search_knn_vector_3d(pointt, 4)
    #check black point's neighbours
    arr = [0,0,0,0,0,0,0,0]
    for i in np.asarray(pcl.points)[idx[1:], :]:
        #check which segments are around the current black point 
        if (i in np.asarray(segments[0].points)) or  (i in np.asarray(segments[1].points)) :
            print("segment0")
            arr[0] = 1
        if (i in np.asarray(segments[2].points)) or (i in np.asarray(segments[3].points)):
            print("segment1")
            arr[1] =  1
        if (i in np.asarray(segments[4].points)) or (i in np.asarray(segments[5].points)):
            print("segment2")
            arr[2] =  1
        if (i in np.asarray(segments[6].points)) or (i in np.asarray(segments[7].points)):
            print("segment3")
            arr[3] =  1
        if (i in np.asarray(segments[8].points)) or (i in np.asarray(segments[9].points)):
            print("segment4")
            arr[4] =  1
    print("---------------------------------------------")
    #completed the process for one black point
    arr[5] = pointt[0]
    arr[6] = pointt[1]
    arr[7] = pointt[2]
    #add processed black point neighbour's info to dic      
    dic.append(arr) 

x = len(dic)
    
arr = []
#compare each black points with other black point

visited = [None] * len(dic)
  
for i in range(0,len(dic)):
    visited[i] = False

for i in range(0,len(dic)):
    point_ = []
    
    if not visited[i]:
        visited[i] = True
        print(dic[i])
        point_.append([dic[i][5],dic[i][6],dic[i][7]])
    for a in range(i+1,len(dic)):
        if not visited[a]:
            if (dic[i][0]==dic[a][0]) & (dic[i][1]==dic[a][1]) & (dic[i][2]==dic[a][2]) & (dic[i][3]==dic[a][3]) & (dic[i][4]==dic[a][4]):
                point_.append([dic[a][5],dic[a][6],dic[a][7]])
                visited[a] = True
            
    #holds black points have same neighbours
    arr.append(point_)


    
#paint each different segments 
x = 0
y = 0
#cl = [[0,1,0],[1,0,0],[0,0,1]]
# i is segment
for i in arr: 
    # a is black point
    for a in i:
        rest.points[x] = a
        rest.colors[x] = cl[y]
        x += 1
    #print(cll[y])
    y +=1
    if y == 10:
        y=0



o3d.visualization.draw_geometries([rest])

 
