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
pcd = SOURCE_PCD.sample_points_poisson_disk(number_of_points = 500)
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
max_plane_idx=20
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


print("colors")
print(np.asarray(list(colors[:3])))
#o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])

o3d.visualization.draw_geometries([pcl])

distances = np.max(rest.compute_nearest_neighbor_distance())
print("dist")
print(distances)
nb_neighbors = 5
#o3d.visualization.draw_geometries([rest])




num_of_points = len(np.asarray(rest.points))


pcd_tree = o3d.geometry.KDTreeFlann(pcl)
#paint first point to red
#rest.colors[6] = [1, 0, 0]
#[k, idx, _] = pcd_tree.search_knn_vector_3d(rest.points[6], 16)
#np.asarray(pcl.colors)[idx[1:], :] = [0, 1, 0]

 

distances = np.max(rest.compute_nearest_neighbor_distance())


arr_ = []
#holds data of each black points neighbours
dic = []
dic2 = []

#holds black points processing
dic_point = []

a = 0
pointt = rest.points[0]
stack = []
arr_stack = []

stack.append(pointt)

#add arr_stack to check current black point processed before 
arr_stack.append(pointt)

k = []
a = 1
#check all black points until stack is empty
while stack:
    pointt = stack.pop()
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pointt, distances)
    #check black point's neighbours
    arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in np.asarray(pcl.points)[idx[1:], :]:
        #if black point's neighbour is also black
        if i in np.asarray(rest.points):
            #check if it is not processed before
            if i not in np.asarray(arr_stack):
                #add to stack to process
                arr_stack.append(i)
                print("Black point" + str(i))
                stack.append(i)
        #check which many segments do current black point have around
        if i in np.asarray(segments[0].points):
            print("segment0 neighbour of point :" + str(i))
            arr[0] = arr[0] + 1
        if i in np.asarray(segments[1].points):
            print("segment1")
            arr[1] = arr[1] + 1
            print("segment1 neighbour of point :" + str(i))
        if i in np.asarray(segments[2].points):
            print("segment2")
            arr[2] = arr[2] + 1
        if i in np.asarray(segments[3].points):
            print("segment3")
            arr[3] = arr[3] + 1
        if i in np.asarray(segments[4].points):
            print("segment4")
            arr[4] = arr[4] + 1
        if i in np.asarray(segments[5].points):
            print("segment5")
            arr[5] = arr[5] + 1
        if i in np.asarray(segments[6].points):
            print("segment6")
            arr[6] = arr[6] + 1
        if i in np.asarray(segments[7].points):
            print("segment7")
            arr[7] = arr[7] + 1
        if i in np.asarray(segments[8].points):
            print("segment8")
            arr[8] = arr[8] + 1
        if i in np.asarray(segments[9].points):
            print("segment9")
            arr[9] = arr[9] + 1
        if i in np.asarray(segments[10].points):
            print("segment10")
            arr[10] = arr[10] + 1
        if i in np.asarray(segments[11].points):
            print("segment11")
            arr[11] = arr[11] + 1
        if i in np.asarray(segments[12].points):
            print("segment12")
            arr[12] = arr[12] + 1
        if i in np.asarray(segments[13].points):
            print("segment13")
            arr[13] = arr[13] + 1
        if i in np.asarray(segments[14].points):
            print("segment14")
            arr[14] = arr[14] + 1
        if i in np.asarray(segments[15].points):
            print("segment15")
            arr[15] = arr[15] + 1
        if i in np.asarray(segments[16].points):
            print("segment16")
            arr[16] = arr[16] + 1
        if i in np.asarray(segments[17].points):
            print("segment17")
            arr[17] = arr[17] + 1
        if i in np.asarray(segments[18].points):
            print("segment18")
            arr[18] = arr[18] + 1
        if i in np.asarray(segments[19].points):
            print("segment19")
            arr[19] = arr[19] + 1
            
    #completed the process for one black point
        
    #add processed black point neighbour's info to dic      
    dic.append(arr) 
    dic2.append(arr)
    #add processed black point to dic_point
    dic_point.append(pointt)
    
    #if there are no black neighbour araound current black point and there is still black point to trace
    if (not stack) & (len(arr_stack) != (len(rest.points))):
        pointt = rest.points[a]
        #find an next unprocessed black point to add stack 
        while (pointt in np.asarray(arr_stack)) & (a != (len(rest.points) - 1) ):
            pointt = rest.points[a]
            a = a + 1
        #add next unprocessed black point to stack
        if pointt not in np.asarray(arr_stack):
            arr_stack.append(pointt)
            stack.append(pointt)

      


#print(np.asarray(pcd.colors)[idx[:], :])
#n = np.asarray(pcd.colors)[idx[:], :][1] 

print("---")
print(len(dic))
print(len(rest.points))
arr = []

#compare each black point's information with other information
for i in dic:
    point_indexx = []
    #remove black point's information in comparison
    dic.remove(i)
    #get index of removed information
    indd = dic2.index(i)
    #add to point_index which holds black points index in same comparison
    point_indexx.append(indd)
    if not dic:
        break
    
    for a in dic:
        in0 = abs(i[0]-a[0])
        in1 = abs(i[1]- a[1])
        in2 = abs(i[2]-a[2])
        in3 = abs(i[3]- a[3])
        in4 = abs(i[4]-a[4])
        in5 = abs(i[5]- a[5])
        in6 = abs(i[6]-a[6])
        in7 = abs(i[7]- a[7])
        in8 = abs(i[8]-a[8])
        in9 = abs(i[9]- a[9])
        in10 = abs(i[10] -a[10])
        in11 = abs(i[11]-a[11])
        in12 = abs(i[12]- a[12])
        in13 = abs(i[13]-a[13])
        in14 = abs(i[14]- a[14])
        in15 = abs(i[15]-a[15])
        in16 = abs(i[16]- a[16])
        in17 = abs(i[17]-a[17])
        in18 = abs(i[18]- a[18])
        in19 = abs(i[19]-a[19])
        #if there is at most 2 points more or less for each segment around them then black points will be same segment
        if (in0<1) & (in1<1) & (in2<1) & (in3<1) & (in4<1) & (in5<1) & (in6<1) & (in7<1) & (in8<1) & (in9<1) & (in10<1) & (in11<1) & (in12<1) & (in13<1) & (in14<1) & (in15<1) & (in16<1) & (in17<1) & (in18<1) & (in19<1):
            dic.remove(a)
            indd = dic2.index(a)
            point_indexx.append(indd)
            
    #holds black points index on same segments
    arr.append(point_indexx)


#paint each different segments 
cl = [[0,1,0],[1,0,0],[0,0,1]]
x = 0
y = 0

# i is segment
for i in arr: 
    # a is black point index
    for a in i:
        rest.points[x] = dic_point[a]
        if y < len(cl):
            rest.colors[x] = cl[y]
        x += 1
    #print(cll[y])
    y +=1
    if y == 3:
        y=0
        


o3d.visualization.draw_geometries([rest])


