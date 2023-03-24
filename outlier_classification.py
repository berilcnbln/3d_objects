import open3d as o3d
import numpy as np
import scipy as sc
import copy
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
import itertools


#DBSCAN Clustering
SOURCE_PCD =  o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/9.ply")
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
    
#End of dbscan clustering

pcl += rest

#pcl is whole point cloud
o3d.visualization.draw_geometries([pcl])

#rest is only black points
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
    #arr holds segment(color) information of neighbours around black point, last 3 index holds current black point coordinates
    arr = [0] * (int(max_plane_idx/2)+3)
    #find nearest points to black point
    #d = ((np.asarray(pcl.points)-pointt)**2).sum(axis=1) 
    #ndx = d.argsort() 
    #find first 30 neighbours of current black point
    #for i in np.asarray(pcl.points)[ndx[:30]]:
    
    #find neighbour points of black point inside the circle 
    #distance is max distance between points in whole point cloud as a radius
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pointt, distances)
    for i in np.asarray(pcl.points)[idx[1:], :]:
        plane_idx = 0
        arr_idx = 0
        while plane_idx != (max_plane_idx):
            #check which segments are around the current black point 
            #In dbscan clustering two color are in one segment like open blue and dark blue are looking as one cluster
            if (i in np.asarray(segments[plane_idx].points)) or  (i in np.asarray(segments[plane_idx+1].points)) :
                print("segment" + str(plane_idx))
                #arr_idx means segment(color)
                arr[arr_idx] = 1
            plane_idx += 2
            arr_idx += 1
            
    print("----------------------")
    #completed the process for one black point
    arr[arr_idx] = pointt[0]
    arr[arr_idx + 1] = pointt[1]
    arr[arr_idx + 2] = pointt[2]
    
    #add black point neighbour's information to dictionary      
    dic.append(arr)
 

arr_ = []
visited = [False] * len(dic)
n = 0

for i in range(0,len(dic)):
    #point_ is cluster array, holds black points have same color neighbours
    point_ = []
    if visited[i]== False:
        visited[i] = True
        n = 0
        #add current black point to cluster
        point_.append([dic[i][int((len(arr)-3))],dic[i][int((len(arr)-2))],dic[i][int((len(arr)-1))]])
    
        #compare black point's neighbours segment information with other black points
        for a in range(i+1,len(dic)):
            if visited[a]== False:
                true_all = 0
                
                for dic_ind in range(0,(len(arr)-3)):
                    if (dic[i][dic_ind]==dic[a][dic_ind]):
                        true_all += 1
                
                if true_all == (len(arr)-3):
                    #if current black point segment (color) information are same with other black point, add other black point to cluster
                    point_.append([dic[a][int((len(arr)-3))],dic[a][int((len(arr)-2))],dic[a][int((len(arr)-1))]])
                    visited[a] = True
                    n = 1
                
        
        if n==1:
            #add cluster array to arr_
            arr_.append(point_)

            

x = 0
y = 0
rest.paint_uniform_color([1, 1, 1])


for i in arr_: 
    # a is black point in cluster array
    for a in i:
        rest.points[x] = a
        #cl holds different colors
        rest.colors[x] = cl[y]
        x += 1
        
    y +=1
    if y == max_plane_idx:
        y=0
        


o3d.visualization.draw_geometries([rest])
