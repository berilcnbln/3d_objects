# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import matplotlib.pyplot as plt


# This example displays a point cloud and if you Ctrl-click on a point
# (Cmd-click on macOS) it will show the coordinates of the point.
# This example illustrates:
# - custom mouse handling on SceneWidget
# - getting a the depth value of a point (OpenGL depth)
# - converting from a window point + OpenGL depth to world coordinate


class ExampleApp:
    o3d.visualization.webrtc_server.enable_webrtc()
    world = ""


    def __init__(self, cloud):
        # We will create a SceneWidget that fills the entire window, and then
        # a label in the lower left on top of the SceneWidget to display the
        # coordinate.
        app = gui.Application.instance

        self.window = app.create_window("Open3D - GetCoord Example", 1024, 768)
        # Since we want the label on top of the scene, we cannot use a layout,
        # so we need to manually layout the window's children.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.picked_points = []

        self.window.add_child(self.info)

        self.info2 = gui.Label("")
        self.info2.visible = True


        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.cloud = cloud
        self.segments = {}
        self.max_plane_idx = 0
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        mat.point_size = 3 * self.window.scaling
        if len(self.picked_points) != 0:
            cloud.paint_uniform_color([1, 0, 0])

        def show():
            self.button.text = "Clustering Mode On        Please Choose a Cluster              "

            self.cloud.paint_uniform_color([0, 0, 1])
            pcd = self.cloud
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
                                 fast_normal_computation=True)
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)
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
            segment_models = {}
            segments2 = {}
            max_plane_idx2 = 5
            rest = pcd

            for i in range(max_plane_idx2):
                colors = plt.get_cmap("tab20")(i)
                segment_models[i], inliers = rest.segment_plane(
                    distance_threshold=0.01, ransac_n=3, num_iterations=10000)
                segments2[i] = rest.select_by_index(inliers)
                print("points")
                print(segments2[i])
                segments2[i].paint_uniform_color(list(colors[:3]))
                self.cloud = self.cloud + segments2[i]
                rest = rest.select_by_index(inliers, invert=True)
                print("pass", i, "/", max_plane_idx2, "done.")



            self.segments = segments2
            self.max_plane_idx = max_plane_idx2
            self.widget3d.scene.add_geometry("Point Cloud2", self.cloud, mat)




        self.button = gui.Button("             Clustering Mode           ")
        self.button.enabled = True

        self.button.visible = True
        self.button.set_on_clicked(show)

        self.window.add_child(self.button)


        self.widget3d.scene.add_geometry("Point Cloud", self.cloud, mat)



        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        #self.widget3d.look_at(center, [0, 0, 10], [0, 1, 0])


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
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        #if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                #gui.KeyModifier.SHIFT):



        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                    world = []
                    world = self.widget3d.scene.camera.unproject(
                        event.x, self.widget3d.frame.height-event.y, 0.99, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])
                    n = [world[0], world[1], world[2]]
                    print("Point Selected: ")

                    print(world[0], world[1], world[2])
                    self.picked_points.append(n)

                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, self.widget3d.frame.height-event.y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])
                    n = [world[0], world[1], world[2]]
                    print("Point Selected: ")
                    print(world[0], world[1], world[2])
                    self.picked_points.append(n)


                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    n = [world[0], world[1], world[2]]
                    d = ((np.asarray(self.cloud.points) - n) ** 2).sum(axis=1)
                    ndx = d.argsort()
                    cluster_no = []
                    print("closest points")
                    print(np.asarray(self.cloud.points)[ndx[:5]])
                    if len(self.picked_points) != 0:
                        ind_segment = 0

                        for ind_segment in range(0, self.max_plane_idx):
                            if np.asarray(self.cloud.points)[ndx[:5]][0] in np.asarray(self.segments[ind_segment].points):
                                print(ind_segment)
                                break

                        if ind_segment not in cluster_no:
                            cluster_no.append(ind_segment)
                            if ind_segment < len(self.segments):
                                self.segments[ind_segment].paint_uniform_color([0, 0, 1])
                                cloud_ = self.cloud + self.segments[ind_segment]

                                mat = rendering.MaterialRecord()
                                mat.shader = "defaultUnlit"
                                # Point size is in native pixels, but "pixel" means different things to
                                # different platforms (macOS, in particular), so multiply by Window scale
                                # factor.
                                mat.point_size = 3 * self.window.scaling
                                print("dönüşme kısmı")
                                self.widget3d.scene.add_geometry(f"point {len(np.asarray(cloud_.points))}",
                                                                 cloud_, mat)


                    # We are sizing the info label to be exactly the right size,
                    # so since the text likely changed width, we need to
                    # re-layout to set the new frame.
                    self.window.set_needs_layout()


                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED



app = gui.Application.instance
app.initialize()


SOURCE_PCD = o3d.io.read_triangle_mesh(r"/Users/beril/PycharmProjects/pythonProject2/files/8.ply")
cloud = SOURCE_PCD.sample_points_poisson_disk(number_of_points=10000)
ex = ExampleApp(cloud)

print(ex.world)
app.run()

