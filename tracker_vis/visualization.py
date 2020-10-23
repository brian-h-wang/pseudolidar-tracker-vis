from tracker_vis.tracking import TrackerBoundingBox, TrackerResults
import numpy as np
from pathlib import Path
import open3d as o3d
import os
import time

def cam_to_velo_frame(velo_points):
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    return R.dot(velo_points)

class TrackingVisualizer(object):

    def __init__(self, results_path, pointcloud_path, fps=60):
        self.tracking_results = TrackerResults.load(Path(results_path))
        self.pointcloud_path = Path(pointcloud_path)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()

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

        # Set viewpoint to camera position
        vc = self.vis.get_view_control()
        vc.set_up(np.array([0, -1, 0]))
        vc.set_front(np.array([0, 0, -1]))


    def visualize_all(self):
        t_prev_frame = time.time()
        n_frames = self.tracking_results.n_frames
        frame = 0

        while True:
            t = time.time()
            if t - t_prev_frame >= (1.0 / self.fps):
                self.visualize_frame(frame)
                frame += 1
                if frame >= n_frames:
                    break
                t_prev_frame = t
            self.update_vis()


    def visualize_frame(self, frame):
        # Load points as a numpy array
        points = np.fromfile(self.pointcloud_path / ("%06d.bin" % frame), dtype=np.float32)
        points = (points.reshape((-1, 4))[:,0:3])
        points = cam_to_velo_frame(points.T).T
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.vis.update_geometry(self.pcd)

        bboxes = [bbox.to_o3d() for bbox in self.tracking_results[frame]]
        if len(bboxes) > 0:
            print("%d boxes at time %d" % (len(bboxes), frame))

        for prev_bbox in self.prev_bboxes:
            self.vis.remove_geometry(prev_bbox, reset_bounding_box=False)
        for bbox in bboxes:
            self.vis.add_geometry(bbox, reset_bounding_box=False)
        self.prev_bboxes = bboxes



    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.close()

