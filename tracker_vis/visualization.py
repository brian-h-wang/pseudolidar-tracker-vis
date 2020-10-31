from tracker_vis.tracking import TrackerBoundingBox, TrackerResults
import numpy as np
from pathlib import Path
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from skimage.io import imread, imshow, imread_collection
import matplotlib.pyplot as plt

GT_BOX_COLOR = [0.4, 0.95, 0.3] # bright green
DET_COLOR = [0.95, 0.3, 0.3] # bright red

def cam_to_velo_frame(velo_points):
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    return R.dot(velo_points)

class TrackingVisualizer(object):

    def __init__(self, pointcloud_path, results_path=None, gt_path=None, detections_path=None, image_path=None,
                 fps=60,  n_skip=1):
        self.tracking_results = self.detections = self.ground_truth = None
        if results_path is not None:
            self.tracking_results = TrackerResults.load(Path(results_path))
        if detections_path is not None:
            self.detections = TrackerResults.load_from_detections(Path(detections_path), box_color=DET_COLOR)
        if gt_path is not None:
            self.ground_truth = TrackerResults.load(Path(gt_path), box_color=GT_BOX_COLOR)
        self.pointcloud_path = Path(pointcloud_path)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=720, width=960)
        self.pcd = o3d.geometry.PointCloud()

        self.show_images = image_path is not None
        if self.show_images:
            self.image_path = Path(image_path)
            self.images = imread_collection(str(self.image_path / "*.png"), conserve_memory=False)
            self.image_vis = o3d.visualization.Visualizer()
            self.image_vis.create_window(height=720, width=720, left=1024)

        # self.viewer = ImageViewer(self.images[0])
        # self.viewer.show()


        # Load the first point cloud
        # Without doing this, the point cloud colors and visualizer zoom are weird
        points = np.fromfile(self.pointcloud_path / ("%06d.bin" % 0), dtype=np.float32)
        points = (points.reshape((-1, 4))[:,0:3])
        points = cam_to_velo_frame(points.T).T
        self.pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        self.prev_bboxes = []
        self.fps = fps
        if n_skip < 1:
            n_skip = 1
        self.n_skip = n_skip

        # Adjust render options
        render_option: o3d.visualization.RenderOption = self.vis.get_render_option()
        # render_option.background_color = [0.005, 0.005, 0.005]
        render_option.background_color = [1.0, 1.0, 1.0]
        # render_option.point_size = 3.0
        render_option.point_size = 0.05

        # Set viewpoint to camera position
        vc = self.vis.get_view_control()
        vc.set_up(np.array([0, -1, 1]))
        vc.set_front(np.array([0, -0.5, -1]))
        vc.set_lookat([0, 0, 5])
        vc.set_zoom(0.1)

        # For making scatter plot of bounding box ranges
        self.box_ranges = []
        self.box_time_steps = []


    def visualize_all(self):
        t_prev_frame = time.time()
        if self.tracking_results:
            n_frames = self.tracking_results.n_frames
        elif self.ground_truth:
            n_frames = self.ground_truth.n_frames
        elif self.detections:
            n_frames = self.detections.n_frames
        frame = 0

        while True:
            t = time.time()
            if t - t_prev_frame >= (1.0 / self.fps):
                # print("\r[Frame %d]" % (frame), end='')
                print("[Frame %d]" % (frame))
                self.visualize_frame(frame)
                frame += self.n_skip
                if frame >= n_frames:
                    break
                t_prev_frame = t
            self.update_vis()


    def visualize_frame(self, frame):
        # Load points as a numpy array
        points = np.fromfile(self.pointcloud_path / ("%06d.bin" % frame), dtype=np.float32)
        if self.show_images:
            image = self.images[frame]
            img_o3d = o3d.geometry.Image(image)
            self.image_vis.clear_geometries()
            self.image_vis.add_geometry(img_o3d)
            self.image_vis.poll_events()
        points = (points.reshape((-1, 4))[:,0:3])
        points = cam_to_velo_frame(points.T).T
        self.pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.ones(points.shape) * 0.5

        bboxes = []
        if self.tracking_results is not None:
            bboxes += self.tracking_results[frame]
        if self.ground_truth is not None:
            bboxes += self.ground_truth[frame]
        if self.detections is not None:
            bboxes += self.detections[frame]

        bboxes_o3d = [bbox.to_o3d() for bbox in bboxes]

        for prev_bbox in self.prev_bboxes:
            self.vis.remove_geometry(prev_bbox, reset_bounding_box=False)
        for bbox in bboxes_o3d:
            self.vis.add_geometry(bbox, reset_bounding_box=False)
            # in_box = bbox.get_point_indices_within_bounding_box(self.pcd.points)
            # colors[in_box,:] = bbox.color
        self.prev_bboxes = bboxes_o3d

        if self.tracking_results is not None:
            self.box_ranges += [bbox.range for bbox in self.tracking_results[frame]]
            self.box_time_steps += [frame for _ in self.tracking_results[frame]]

        # self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)

    def plot_ranges(self):
        plt.plot(self.box_time_steps, self.box_ranges, '.')
        plt.xlabel("Time step")
        plt.ylabel("Bounding box range")
        plt.show()


    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.close()

