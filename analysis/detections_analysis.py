import numpy as np
from tracker_vis.tracking import TrackerResults, TrackerBoundingBox
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class DetectionsHistogram(object):

    def __init__(self, detections, ground_truth, hist_bin_size=1.0, hist_range_max=15.0, assoc_threshold=1.0, max_frame=2000):
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

        for frame in range(min(ground_truth.n_frames, max_frame)):
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

        self.x_errors = x_errors
        self.z_errors = z_errors
        self.size_errors = size_errors

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
        self.bin_ranges = np.arange(n_bins) * hist_bin_size
        self.bin_size = hist_bin_size
        self.bin_max_range = hist_range_max

def plot_means_stddevs(det_hist, trk_hist=None, max_range=None, width=0.6):
    """
    Plot the means and std deviations of X, Y, and size,
    among the detections matched to a GT instance.

    Parameters
    ----------
    det_hist: DetectionsHistogram
    trk_hist: DetectionsHistogram
    max_range

    """
    # Determine which bins to plot based on the max range
    if max_range is None:
        max_range = det_hist.bin_max_range
    bins = [b for b in det_hist.bin_ranges if b < max_range]
    n_bins = len(bins)

    plt.rcParams["axes.titleweight"] = "bold"

    # Plot means
    fig, axs = plt.subplots(3,2)
    fig.tight_layout()
    ax_mean_x: plt.Axes = axs[1,0]
    ax_mean_z: plt.Axes = axs[0,0]
    ax_mean_size: plt.Axes = axs[2,0]
    ax_std_x: plt.Axes = axs[1,1]
    ax_std_z : plt.Axes= axs[0,1]
    ax_std_size : plt.Axes= axs[2,1]

    ax_all = [ax_mean_x, ax_mean_z, ax_mean_size, ax_std_x, ax_std_z, ax_std_size]
    for ax in ax_all:
        ax.set_xlabel("Range from camera [m]")
        ax.set_xticks(bins)
    for ax in [ax_mean_x, ax_mean_z, ax_mean_size]:
        ax.set_ylabel("Mean absolute error [m]")
    for ax in [ax_std_x, ax_std_z, ax_std_size]:
        ax.set_ylabel("Std. dev. of error [m]")

    # bins for detections and trk
    # Resizes bar width depending on if we're plotting detections and tracker, or dets only
    bins = np.array(bins)
    if trk_hist is None:
        b_det = bins
        b_trk = bins
    else:
        width = width/2.
        b_det = bins - width/2
        b_trk = bins + width/2

    axs = [ax_mean_x, ax_mean_z, ax_mean_size, ax_std_x, ax_std_z, ax_std_size]
    det_data = [det_hist.x_means, det_hist.z_means, det_hist.size_means,
                det_hist.x_stddevs, det_hist.z_stddevs, det_hist.size_stddevs]
    trk_data = [trk_hist.x_means, trk_hist.z_means, trk_hist.size_means,
                trk_hist.x_stddevs, trk_hist.z_stddevs, trk_hist.size_stddevs]

    variables = ["x-position", "y_position", "tree size"]
    titles = ["Mean error of %s" % v for v in variables] + \
        ["Standard deviation of %s error" % v for v in variables]

    for i in range(6):
        ax = axs[i]
        det = det_data[i][:n_bins]
        trk = trk_data[i][:n_bins]

        ax.bar(b_det, det, width=width, label="Raw detections")
        if not trk_hist is None:
            ax.bar(b_trk, trk, width=width, label="Tracker results")
        ax.legend(loc='upper left')
        ax.set_ylim([0, 0.6])
        ax.set_title(titles[i])

    plt.show()

def plot_precision_recall(det_hist, trk_hist=None, max_range=None):
    """

    Parameters
    ----------
    det_hist: DetectionsHistogram
    trk_hist: DetectionsHistogram
    max_range

    """
    # Determine which bins to plot based on the max range
    if max_range is None:
        max_range = det_hist.bin_max_range
    bins = [b for b in det_hist.bin_ranges if b < max_range]
    n_bins = len(bins)

    # Plot means
    fig, axs = plt.subplots(2,1)
    fig.tight_layout()
    ax_mean_p: plt.Axes = axs[0]
    ax_mean_r: plt.Axes = axs[1]
    plt.rcParams["axes.titleweight"] = "bold"

    ax_all = [ax_mean_p, ax_mean_r]
    for ax in ax_all:
        ax.set_xlabel("Range from camera [m]")
        ax.set_xticks(bins)

    det_p = np.zeros(n_bins)
    for i in range(n_bins):
        matched = det_hist.matched_detections_count[i]
        fp = det_hist.false_positives_count[i]
        if matched + fp == 0:
            det_p[i] = 0
        else:
            det_p[i] = matched / (matched + fp)
    det_p = det_p[0:n_bins]
    det_r = det_hist.matched_detections_count / (det_hist.matched_detections_count + det_hist.false_negatives_count)
    det_r = det_r[0:n_bins]

    ax_mean_p.bar(bins, det_p, width=1.0, label="Raw detections")
    ax_mean_p.legend(loc='upper left')
    ax_mean_p.set_ylim([0, 1.0])
    ax_mean_p.set_title("Precision of detections vs. tracker results")
    ax_mean_p.set_ylabel("Precision")

    ax_mean_r.bar(bins, det_r, width=1.0, label="Raw detections")
    ax_mean_r.legend(loc='upper left')
    ax_mean_r.set_ylim([0, 1.0])
    ax_mean_r.set_title("Recall of detections vs. tracker results")
    ax_mean_r.set_ylabel("Recall")

    plt.show()

def plot_fp_fn_counts(det_hist, trk_hist=None, max_range=None, width=0.6):
    """

    Parameters
    ----------
    det_hist: DetectionsHistogram
    trk_hist: DetectionsHistogram
    max_range

    """
    min_cutoff = 1
    # Determine which bins to plot based on the max range
    if max_range is None:
        max_range = det_hist.bin_max_range
    bins = [b for b in det_hist.bin_ranges if b < max_range]
    # bins = bins[min_cutoff:]
    bins = bins[min_cutoff:]
    n_bins = len(bins)

    # Plot means
    fig, axs = plt.subplots(3,1)
    fig.tight_layout()
    plt.rcParams["axes.titleweight"] = "bold"

    ylabels = ["Associated detections", "False positives", "False negatives"]
    titles = ["Number of bounding boxes matched with ground truth trees vs. range", "Number of false positives vs. range",
              "Number of false negatives vs. range"]

    data_det = [det_hist.matched_detections_count, det_hist.false_positives_count, det_hist.false_negatives_count]
    data_trk = [trk_hist.matched_detections_count, trk_hist.false_positives_count, trk_hist.false_negatives_count]

    bins = np.array(bins)
    if trk_hist is None:
        b_det = bins
        b_trk = bins
    else:
        width = width/2.
        b_det = bins - width/2
        b_trk = bins + width/2

    for i in range(3):
        # ax 0: count of matched detections
        # ax 1: count of false positivesj
        # ax 2: count of false negatives
        ax = axs[i]
        plt.sca(ax)
        ax.set_xlabel("Range from camera [m]")


        plt.xticks(ticks=bins, labels=["%dm - %dm" % (b, b+det_hist.bin_size) for b in bins], fontsize=8)
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        y_det = data_det[i][min_cutoff:min_cutoff+n_bins]
        y_trk = data_trk[i][min_cutoff:min_cutoff+n_bins]
        bd = ax.bar(b_det, y_det, width=width, label="Raw detections")
        bt = ax.bar(b_trk, y_trk, width=width, label="Tracker estimate means")
        ax.legend(loc='upper right')
        # ax_mean_p.set_ylim([0, 1.0])
        ax.set_ylim([0, 2000])

        for i in range(n_bins):
            # Text on the top of each barplot
            ax.text(x=b_det[i], y=y_det[i] + 0.2, s=str(y_det[i]), size=8,
                    color=u'#1f77b4', horizontalalignment='center')
            ax.text(x=b_trk[i], y=y_trk[i] + 0.2, s=str(y_trk[i]), size=8,
                    color=u'#ff7f0e', horizontalalignment='center')

    plt.show()


def get_bin_index(bbox, hist_bin_size):
    r = bbox.range
    return int(np.floor(r / hist_bin_size))










