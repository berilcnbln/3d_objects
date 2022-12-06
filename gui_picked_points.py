import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import matplotlib.pyplot as plt

world = []
class ExampleApp:

    def __init__(self, cloud):

        app = gui.Application.instance
        self.window = app.create_window("Open3D - GetCoord Example", 1024, 768)

        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)
        self.picked_points = []
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"

        self.mat.point_size = 3 * self.window.scaling

        if len(self.picked_points) != 0:
            print("x")
            cloud.paint_uniform_color([1, 0, 0])
            
            
        self.widget3d.scene.add_geometry("Point Cloud", cloud, self.mat)

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _on_mouse_widget3d(self, event):

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):
            def depth_callback(depth_image):

                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                    np.asarray(cloud.colors)[:] = [0,0,1]
                    self.widget3d.scene.add_geometry("Point Cloud", cloud, self.mat)
                    world = []
                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, (self.widget3d.frame.height - event.y), depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])
                    
                n = [world[0], world[1], world[2]]
                
                ind = np.where(np.asarray(cloud.points) == n)    
                print(ind)    
                self.picked_points.append(n)
                print(world)
                print(n)
                
               

                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")

                    self.window.set_needs_layout()

                    mat = o3d.visualization.rendering.MaterialRecord()
                    mat.shader = "defaultLit"

                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    sphere.paint_uniform_color([0.1, 0.1, 0.7])
                    sphere.compute_vertex_normals()
                    n = [world[0], world[1], world[2]]
                    d = ((np.asarray(cloud.points)-n)**2).sum(axis=1) 
                    ndx = d.argsort() 
                    print("closest points")
                    print(np.asarray(cloud.points)[ndx[:5]])
                    if len(self.picked_points) != 0:
                        ind_segment = 0

                        for ind_segment in range (0, max_plane_idx):
                            if np.asarray(cloud.points)[ndx[:5]][0] in np.asarray(segments[ind_segment].points):
                                break
                                
                        if ind_segment not in cluster_no:
                            cluster_no.append(ind_segment)    
                            segments[ind_segment].paint_uniform_color([0, 0, 1])
                            cloud_ = cloud + segments[ind_segment]
                            self.widget3d.scene.add_geometry(f"point {len(np.asarray(cloud_.points))}", cloud_, self.mat)
                            bounds = self.widget3d.scene.bounding_box
                            center = bounds.get_center()
                            self.widget3d.setup_camera(60, bounds, center)
                            self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])


                        print("y")
                    #sphere.translate(np.asarray(cloud.points)[ndx[:5]][0])
                    #self.widget3d.scene.add_geometry(f"point {len(self.picked_points)}", sphere, mat)

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED



app = gui.Application.instance
app.initialize()


SOURCE_PCD = o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/8.ply")
# ---------------------------------------------------------------------------------
cloud = SOURCE_PCD.sample_points_poisson_disk(number_of_points=100000)
cloud.paint_uniform_color([0, 0, 1])
pcd = cloud
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
max_plane_idx=5
rest=pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold = 0.01,ransac_n = 3,num_iterations=10000)
    segments[i] = rest.select_by_index(inliers)
    print("points")
    print(segments[i])
    segments[i].paint_uniform_color(list(colors[:3]))
    cloud = cloud + segments[i]
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

cluster_no = []

ex = ExampleApp(cloud)

app.run()


