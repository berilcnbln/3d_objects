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



SOURCE_PCD = o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/5.ply")
TARGET_PCD = o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject4/sphere1.ply")

x, y, z = SOURCE_PCD.get_center().T
SOURCE_PCD = copy.deepcopy(SOURCE_PCD).translate((-x, -y, -z))
SOURCE_PCD.scale(1, center=(0, 0, 0))

x, y, z = TARGET_PCD.get_center().T
TARGET_PCD = copy.deepcopy(TARGET_PCD).translate((-x, -y, -z))
TARGET_PCD.scale(1, center=(0, 0, 0))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    # transformation time
    start_time = time.time()
    source_temp.transform(transformation)
    t_translation = time.time() - start_time
    o3d.visualization.draw_geometries([source_temp, target_temp])


pcd_source = SOURCE_PCD.sample_points_poisson_disk(number_of_points=500, init_factor=5)
pcd_target = TARGET_PCD.sample_points_poisson_disk(number_of_points=500, init_factor=5)

threshold = 10000
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

draw_registration_result(pcd_source, pcd_target, trans_init)
#o3d.visualization.draw_geometries([n])



print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    pcd_source, pcd_target, threshold, trans_init)
print(trans_init)
print("evaluation")
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_source, pcd_target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print("reg_p2p")
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)

draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)
