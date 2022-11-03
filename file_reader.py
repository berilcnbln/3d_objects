import warnings
warnings.filterwarnings('ignore')
import vapory

import trimesh
import gmsh
import sys
{sys.executable}
from stl import mesh
import point_cloud_utils as pcu
import open3d as o3d
#Initialize gmsh:
import files_open as f
gmsh.initialize()
print(sys.path)
import trimesh

import gmsh
import sys
{sys.executable}
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

arr=[]
strr = ".ply"
if gmsh.isInitialized() :
    for i in f.file:
        if i == '.DS_Store':
            continue
        a = "/Users/beril/Desktop/igs/"
        a += str(i)
        print("converting igs file..")

        mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(file_name =a, gmsh_args = [
                ("Mesh.Algorithm", 3),
                ("Mesh.CharacteristicLengthFromCurvature", 100),
                ("General.NumThreads", 1),
                ("Mesh.MinimumCirclePoints", 32)]))

        v = i.split(".ig")
        val = v[0]
        c = "/Users/beril/Desktop/folder/"
        c += val
        c += strr

        mesh.export(c)
        print("Done exporting ply file")
