import numpy as np
from tracker_vis.tracking import TrackerResults, TrackerBoundingBox
from scipy.optimize import linear_sum_assignment

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

class DetectionsHistogram(object):

    def __init__(self, detections, ground_truth, hist_bin_size=1.0, hist_range_max=15.0, assoc_threshold=3.0):
        """
        Parameters
        ----------
        detections: TrackerResults
        ground_truth: TrackerResults
        hist_bin_size: float
        hist_range_max: float
            Maximum range to consider. The last bin in the histogram will cover detections with a range between
            [hist_range_max - hist_bin_size] and [hist_range_max].
        assoc_threshold: float
            Maximum distance for a detection to be matched with a ground truth instance.

        Returns
        -------

        """
        n_bins = int(hist_range_max / hist_bin_size)

        # Store the errors in x-values, y-values, and detection size, for each detection-GT matching
        x_errors = [[] for _ in range(n_bins)]
        z_errors = [[] for _ in range(n_bins)]
        size_errors = [[] for _ in range(n_bins)]

        # Records the number of matched detections at each range bin
        matched_detections_count = np.zeros(n_bins, dtype=int)

        # Count false positives and negatives at each range bin
        false_positives_count = np.zeros(n_bins, dtype=int)
        false_negatives_count = np.zeros(n_bins, dtype=int)

        for frame in range(ground_truth.n_frames):
            # Filter out any detections or gt boxes that are outside the maximum range for the histogram
            detection_boxes = [d for d in detections[frame] if get_bin_index(d, hist_bin_size) < n_bins]
            gt_boxes = [g for g in ground_truth[frame] if get_bin_index(g, hist_bin_size) < n_bins]

            # Compute cost matrix between detections and gt
            C = np.zeros((len(detection_boxes), len(gt_boxes)))
            for i, detection in enumerate(detection_boxes):
                for j, gt in enumerate(gt_boxes):
                    C[i,j] = detection.distance(gt)

            # Perform data association
            det_indices, gt_indices = linear_sum_assignment(C)

            # Count false positives and false negatives based on difference between # of dets and # of gt boxes
            # This will not capture matchings that are over the max range - count these in the loop
            for det_index, gt_index in zip(det_indices, gt_indices):
                detection = detection_boxes[det_index]
                gt = gt_boxes[gt_index]
                # Check if distance is higher than the threshold - if so, this is an invalid match
                if C[det_index, gt_index] > assoc_threshold:
                    # No match -> add one false positive and one false negative
                    false_positives_count[get_bin_index(detection, hist_bin_size)] += 1
                    false_negatives_count[get_bin_index(gt, hist_bin_size)] += 1
                else:
                    # For a successful match, bin index is based off of the range to the GT
                    bin_index = get_bin_index(gt, hist_bin_size)
                    # Count one successfully matched detection
                    matched_detections_count[bin_index] += 1
                    # Note: x is left/right, z is forward/back error
                    x_err = np.abs(detection.x - gt.x)
                    z_err = np.abs(detection.z - gt.z)
                    size_err = np.abs(detection.size - gt.size)

                    # Add to the histogram bins
                    x_errors[bin_index].append(x_err)
                    z_errors[bin_index].append(z_err)
                    size_errors[bin_index].append(size_err)

            # Count up false positives and false negatives
            # More detections than truth boxes means some detections are false positives
            if len(detection_boxes) > len(gt_boxes):
                for i in range(len(detection_boxes)):
                    # if this detection was matched with a GT - skip
                    if i not in det_indices:
                        bin_index = get_bin_index(detection_boxes[i], hist_bin_size)
                        false_positives_count[bin_index] += 1
            # More truth boxes than detections means there are false negatives
            elif len(detection_boxes) < len(gt_boxes):
                for i in range(len(gt_boxes)):
                    # if this detection was matched with a GT - skip
                    if i not in gt_indices:
                        bin_index = get_bin_index(gt_boxes[i], hist_bin_size)
                        false_negatives_count[bin_index] += 1

        # Account for any bins which are empty - i.e. no matches
        for b in range(n_bins):
            if len(x_errors[b]) == 0:
                x_errors[b] = [0]
            if len(z_errors[b]) == 0:
                z_errors[b] = [0]
            if len(size_errors[b]) == 0:
                size_errors[b] = [0]

        self.x_means = [np.mean(xs) for xs in x_errors]
        self.z_means = [np.mean(zs) for zs in z_errors]
        self.size_means = [np.mean(s) for s in size_errors]

        self.x_stddevs = [np.std(xs) for xs in x_errors]
        self.z_stddevs = [np.std(zs) for zs in z_errors]
        self.size_stddevs = [np.std(s) for s in size_errors]

        self.matched_detections_count = matched_detections_count
        self.false_positives_count = false_positives_count
        self.false_negatives_count = false_negatives_count

        self.n_bins = n_bins


def get_bin_index(bbox, hist_bin_size):
    r = bbox.range
    return int(np.floor(r / hist_bin_size))








