import pandas as pd
import open3d as o3d
import numpy as np
import itertools
import copy
import os
import point_cloud_utils as pcu
import time
import plotly.express as px
import xlsxwriter
import plotly.graph_objects as go



def fn():
    file_list = os.listdir("/Users/beril/PycharmProjects/pythonProject2/files")
    return file_list
file = fn()
print(file)

#lst = list(itertools.product(file, repeat=2))

table = []
matrix = []
rmse_ = []
triangles_ = []


for i in file:
    #if value == 3:
     #   break
    #value = value + 1
    a = "files/"
    a += i
    #read mesh files
    inputt = "16.ply"

    input_ = "files/"
    input_ += inputt
    SOURCE_PCD = o3d.io.read_triangle_mesh(input_)
    TARGET_PCD = o3d.io.read_triangle_mesh(a)


    #Target box Volume
    pcd_target = TARGET_PCD.sample_points_poisson_disk(number_of_points=100, init_factor=5)
    hull, _ = pcd_target.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    box = hull_ls.get_axis_aligned_bounding_box()
    v_target = box.volume()


    #Target box Volume
    pcd_source = SOURCE_PCD.sample_points_poisson_disk(number_of_points=100, init_factor=5)
    hull, _ = pcd_source.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    box = hull_ls.get_axis_aligned_bounding_box()
    v_source = box.volume()




    #calculation of delta values of Target value
    max_b = box.get_max_bound()
    xx, yx, zx = max_b
    min_b = box.get_min_bound()
    xn, yn, zn = min_b

    xdelta = xx - xn
    ydelta = yx - yn
    zdelta = zx - zn



    #scale source mesh and recenter it
    x,y,z = SOURCE_PCD.get_center().T
    SOURCE_PCD = copy.deepcopy(SOURCE_PCD).translate((-x, -y, -z))
    SOURCE_PCD.scale(1, center= (0,0,0))


    #scale target mesh and recenter it
    x,y,z = TARGET_PCD.get_center().T
    TARGET_PCD = copy.deepcopy(TARGET_PCD).translate((-x, -y, -z))
    TARGET_PCD.scale(1, center= (0,0,0))


    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        #transformation time
        start_time = time.time()
        source_temp.transform(transformation)
        t_translation = time.time() - start_time

        #volume ratio calculation
        new_mesh = target_temp + source_temp
        hull, _ = new_mesh.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        box_total = hull_ls.get_axis_aligned_bounding_box()
        v_total = box_total.volume()

        hull, _ = target_temp.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        box_temp = hull_ls.get_axis_aligned_bounding_box()
        v_t = box_temp.volume()

        hull, _ = source_temp.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        box_source = hull_ls.get_axis_aligned_bounding_box()
        v_s = box_source.volume()

        a = v_t - v_s
        b = v_s - v_t
        intersection_volume = v_total - a - b
        in_t = intersection_volume / v_t
        in_s = intersection_volume / v_s

        return t_translation, in_t, in_s


    #transformed figure
    def draw_registration_result2(source, target, transformation, val):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_temp.transform(transformation)


        points = np.asarray(source_temp.points)
        x2, y2, z2 = points.T

        fig = px.scatter_3d(x= x2 / max(x2), y= y2 / max(y2), z= z2 / max(z2))

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                yaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                zaxis=dict(nticks=4, range=[-1.05, 1.05], ), ),
            width=700,
            margin=dict(r=100, l=50, b=50, t=50))


        fig.update_traces(marker=dict(size=1,
                                      line=dict(width=0.1,
                                                color='DarkSlateGrey')), selector=dict(mode='markers'))

        t = "T"
        t += val
        tx= ".png"
        t += tx
        fig.write_image(t)



    #merged figure
    def draw_registration_result3(source, target, transformation, val):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_temp.transform(transformation)
        points = np.asarray(source_temp.points)
        x1, y1, z1 = points.T

        points = np.asarray(target_temp.points)
        x2, y2, z2 = points.T

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
        x=x1/(max(x1)), y=y1/(max(y1)), z=z1/(max(z1)),
        mode='markers',
        marker_color='red'
        ))

        fig.add_trace(go.Scatter3d(
        x=x2/(max(x2)), y=y2/(max(y2)), z=z2/(max(z2)),
        marker_color='blue'
        ))


        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                yaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                zaxis=dict(nticks=4, range=[-1.05, 1.05], ), ),
            width=700,
            margin=dict(r=100, l=50, b=50, t=50))



        # Set options common to all traces with fig.update_traces
        fig.update_traces(mode='markers', marker_line_width=2, marker_size=2)
        t = "merge"
        t += val
        tx = ".png"
        t += tx
        fig.write_image(t)


#-----------------------------------------------------------------------------------------------------------------------
    # sample from mesh files


    pcd_source = SOURCE_PCD.sample_points_poisson_disk(number_of_points=500, init_factor=5)
    pcd_target = TARGET_PCD.sample_points_poisson_disk(number_of_points=500, init_factor=5)

    threshold = 10000
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    #initial allignment
    start_time = time.time()
    evaluation = o3d.pipelines.registration.evaluate_registration(
    pcd_source, pcd_target, threshold, trans_init)
    #point to point ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_source, pcd_target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
    tx = time.time() - start_time

    evaluation = o3d.pipelines.registration.evaluate_registration(
        pcd_source, pcd_target, threshold, trans_init)

    #translation matrix calculation time
    start_time = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    m_ = reg_p2p.transformation
    t_matrix = time.time() - start_time


    # target number
    v = i.split(".")
    val = v[0]

    a = 500


    t_translation = draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)
    draw_registration_result2(pcd_source, pcd_target, reg_p2p.transformation, val)
    draw_registration_result3(pcd_source, pcd_target, reg_p2p.transformation, val)


    rmsee_ = reg_p2p.inlier_rmse
    rmse_.append(rmsee_)



    norm = np.linalg.norm(reg_p2p.transformation, ord=None, axis=None)
    norm1 = np.linalg.norm(reg_p2p.transformation, ord=1, axis=None)
    norm2 = np.linalg.norm(reg_p2p.transformation, ord=2, axis=None)
    w_similar, v_similar = np.linalg.eig(reg_p2p.transformation)
    #add each values generated for each target object to table
    #(target_file[0], rmse[1], total time for icp[2], target_volume[3], target name[4], transformation matrix[5], norm0[6], norm1[7], norm2[8], eigen_values[9], point number[10], deltax[11], deltay[12], deltaz[13], transformation_time[14], transformation matrix calculation[15], volume ratio_target[16], volume ratio_source[17])
    table.append([i, round(rmsee_,2), tx, v_target, val, reg_p2p.transformation, norm, norm1, norm2,w_similar, a, xdelta,ydelta,zdelta, t_translation[0],t_matrix,t_translation[1],t_translation[2]])



#end of the for loop generated icp values for each target objects considering input source

#sorted index array according to rmse values
sort_index = np.argsort(rmse_)


a = 0
arr_ = []
arr2_ = []
arr3_ = []
arr4_ = []
arr5_ = []
arr6_ = []
arr7_ = []
arr8_ = []
arr9_ = []
arr10_ = []
arr11_ = []
arr12_ = []
arr13_ = []
arr14_ = []
arr15_ = []
arr16_ = []
arr17_ = []
arr18_ = []
arr19_ = []
arr20_ = []
arr21_ = []



for i in table:
    if a == 1:
        #draw input object image before end of loop
        x = "files/"
        x += inputt
        SOURCE_PCD = o3d.io.read_triangle_mesh(str(x))
        pcd_source = SOURCE_PCD.sample_points_poisson_disk(number_of_points=50000, init_factor=5)
        points = np.asarray(pcd_source.points)
        x2, y2, z2 = points.T
        fig = px.scatter_3d(x=x2 / max(x2), y=y2 / max(y2), z=z2 / max(z2))

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                yaxis=dict(nticks=4, range=[-1.05, 1.05], ),
                zaxis=dict(nticks=4, range=[-1.05, 1.05], ), ),
            width=700,
            margin=dict(r=100, l=50, b=50, t=50))

        fig.update_traces(marker=dict(size=1,
                                      line=dict(width=0.1,
                                                color='DarkSlateGrey')), selector=dict(mode='markers'))

        fig.write_image("input.png")
        break

    print(table[sort_index[a]])
    arr_.append(table[sort_index[a]])
    #name of object
    str_ = arr_[a][0]
    arr3_.append(str_)

    #rmse value
    arr2_.append(arr_[a][1])

    # volume
    arr4_.append(arr_[a][3])

    # time for icp implementation
    arr5_.append(arr_[a][2])

    #T_image name
    arr6_.append(arr_[a][4])

    arr7_.append(arr_[a][5])
    arr8_.append(arr_[a][6])
    arr9_.append(arr_[a][7])
    arr10_.append(arr_[a][8])
    arr11_.append(arr_[a][9])
    arr12_.append(arr_[a][10])
    arr13_.append(arr_[a][11])
    arr14_.append(arr_[a][12])
    arr15_.append(arr_[a][13])
    arr16_.append(arr_[a][14])
    arr17_.append(arr_[a][15])
    arr20_.append(arr_[a][16])
    arr21_.append(arr_[a][17])


    x = "files/"
    x += str_
    SOURCE_PCD = o3d.io.read_triangle_mesh(str(x))

    start_time = time.time()
    #compute number of triangles
    #intersecting_triangles = np.asarray(
      #  SOURCE_PCD.get_self_intersecting_triangles())
    #triangles_num = len(intersecting_triangles)
    triangle_time = time.time() - start_time
    a2 = 0
    arr18_.append(a2)


    start_time = time.time()
    # compute number of clusters
    #pcds = o3d.io.read_point_cloud(str(x))
    #with o3d.utility.VerbosityContextManager(
    #        o3d.utility.VerbosityLevel.Debug) as cm:
    #    labels = np.array(
    #        pcds.cluster_dbscan(eps=0.005, min_points=1, print_progress=True))

    #max_label = labels.max()
    cl_time = time.time() - start_time
    #arr19_.append((max_label/1000 + 1))
    arr19_.append(a2)



    pcd_source = SOURCE_PCD.sample_points_poisson_disk(number_of_points=50000, init_factor=5)
    points = np.asarray(pcd_source.points)
    x2, y2, z2 = points.T

    fig = px.scatter_3d(x=x2 / max(x2), y=y2 / max(y2), z=z2 / max(z2))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-1.0, 1.0], ),
            yaxis=dict(nticks=4, range=[-1.0, 1.0], ),
            zaxis=dict(nticks=4, range=[-1.0, 1.0], ), ),
        width=700,
        margin=dict(r=100, l=50, b=50, t=50))

    fig.update_traces(marker=dict(size=1,
                                  line=dict(width=0.1,
                                            color='DarkSlateGrey')), selector=dict(mode='markers'))

    #fig.show()
    b = str(a)

    b += ".png"
    print(b)
    fig.write_image(b)

    a = a + 1



