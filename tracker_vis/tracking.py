import numpy as np
import open3d as o3d
import os

class TrackerBoundingBox(object):
    """
    Represents a bounding box in the tracking results.
    """

    def __init__(self, x, y, z, h, w, l, rotation_y, track_id):
        """

        Parameters
        ----------
        x: float
        y: float
        z: float
        h: float
        w: float
        l: float
        """
        self.x = x
        self.y = y
        self.z = z
        self.height = h
        self.width = w
        self.length = l
        self.rotation_y = rotation_y
        self.track_id = track_id
        self.color = [0, 0, 0]

    @staticmethod
    def from_kitti(kitti_str):
        """
        Create a TrackerBoundingBox from KITTI format data.

        Parameters
        ----------
        kitti_str: str
            Object information, including 3D bounding box,
            in KITTI object detection format.

        Returns
        -------
        TrackerBoundingBox

        """
        split = kitti_str.split(' ')
        frame = int(split[0])
        track_id = int(split[1])
        # object_type = split[2]
        # indices 3,4,5,6,7,8 are 'truncated', 'occluded', and the four 2D bounding box coords
        h = float(split[10])
        w = float(split[11])
        l = float(split[12])
        x = float(split[13])
        y = float(split[14])
        z = float(split[15])
        # Ignore rotation_y - will be approx. zero for all bboxes
        rotation_y = float(split[5])
        return frame, TrackerBoundingBox(x=x, y=y, z=z, h=h, w=w, l=l, rotation_y=rotation_y, track_id=track_id)

    @staticmethod
    def from_kitti_detection(kitti_str):
        """
        Create a TrackerBoundingBox from KITTI object detection format data.

        Parameters
        ----------
        kitti_str: str
            Object information, including 3D bounding box,
            in KITTI object detection format.

        Returns
        -------
        TrackerBoundingBox

        """
        split = kitti_str.split(' ')
        # object_type = split[0]
        # indices 3,4,5,6,7,8 are 'truncated', 'occluded', and the four 2D bounding box coords
        h = float(split[8])
        w = float(split[9])
        l = float(split[10])
        x = float(split[11])
        y = float(split[12])
        z = float(split[13])
        # Ignore rotation_y - will be approx. zero for all bboxes
        rotation_y = float(split[14])
        return TrackerBoundingBox(x=x, y=y, z=z, h=h, w=w, l=l, rotation_y=rotation_y, track_id=0)

    def to_o3d(self):
        points = np.zeros((8,3))
        i = 0
        # Note: x, y, z are in camera coords
        for dx in [-self.width/2., self.width/2.]:
            for dy in [0, -self.height]:
                for dz in [-self.length/2., self.length/2]:
                    points[i,:] = [self.x + dx, self.y + dy, self.z + dz]
                    i += 1

        points_v3d = o3d.utility.Vector3dVector(points)
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_v3d)
        bbox_o3d.color = self.color
        return bbox_o3d

    def __str__(self):
        return "--------------------------------------\n" \
               "Bounding box ID: %d\n" \
               "Position (x,y,z): [%.3f, %.3f, %.3f]\n" \
               "Dimensions (h,w,l): [%.3f, %.3f, %.3f]\n" \
               "--------------------------------------" % (self.track_id, self.x, self.y, self.z, self.height, self.width, self.length)


class TrackerResults(object):

    def __init__(self, box_color=None):
        self._results = {}
        self.colors = {}
        self.box_color = box_color

    def __getitem__(self, time_step):
        if time_step not in self._results.keys():
            return []
        else:
            return self._results[time_step]

    def add(self, time_step, tracker_bbox):
        assert type(time_step) == int, "Must give an int time step as input"
        assert type(tracker_bbox) == TrackerBoundingBox, "TrackerResults.add takes in a TrackerBoundingBox input"
        if tracker_bbox.track_id not in self.colors.keys():
            if self.box_color is None:
                new_color = list(np.random.random(3) * 0.7 + 0.2)
                self.colors[tracker_bbox.track_id] = new_color
            else:
                self.colors[tracker_bbox.track_id] = self.box_color
        tracker_bbox.color = self.colors[tracker_bbox.track_id]
        self._results[time_step] = self[time_step] + [tracker_bbox]

    @property
    def n_frames(self):
        return np.max(list(self._results.keys()))

    @staticmethod
    def load(data_path, box_color=None):
        """
        Loads a tracking results file and creates a TrackingResults object,
        which can be used for accessing the tracker data.

        The lines of the input file should contain, in order:
        (Descriptions adapted from KITTI tracking devkit readme)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.

        Parameters
        ----------
        data_path

        Returns
        -------

        """
        results = TrackerResults(box_color=box_color)
        with open(data_path, 'r') as data_file:
            lines = [l.rstrip() for l in data_file.readlines()]
        for line in lines:
            frame, bbox = TrackerBoundingBox.from_kitti(line)
            results.add(frame, bbox)
        return results

    @staticmethod
    def load_from_detections(detections_path):
        """
        Loads a tracking results file and creates a TrackingResults object,
        which can be used for accessing the tracker data.

        The lines of the input file should contain, in order:
        (Descriptions adapted from KITTI tracking devkit readme)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.

        Parameters
        ----------
        data_path

        Returns
        -------

        """
        results = TrackerResults()
        for filename in os.listdir(detections_path):
            frame = int(filename.split('.')[0])
            with open(detections_path / filename, 'r') as detections_file:
                lines = [l.rstrip() for l in detections_file.readlines()]
            for line in lines:
                bbox = TrackerBoundingBox.from_kitti_detection(line)
                results.add(frame, bbox)

        return results


