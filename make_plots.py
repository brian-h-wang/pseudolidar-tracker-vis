from analysis.detections_analysis import *
from tracker_vis.tracking import TrackerResults

if __name__ == "__main__":
    det = TrackerResults.load_from_detections("data/detections_epoch70")
    gt = TrackerResults.load("data/tracking_gt_updated.txt")
    trk = TrackerResults.load("data/tracker_results_final.txt")

    trk_var = TrackerResults.load_with_variance("data/trk_highest_R.txt", "data/variances.txt")

    hist = DetectionsHistogram(det, gt, assoc_threshold=1.0)
    hist_trk = DetectionsHistogram(trk, gt, assoc_threshold=1.0)

    # plot_means_stddevs(hist, hist_trk, max_range=9)

    # TODO make plot of FP, FN, MATCH counts. Show little numbers on top of all the bars.

    # plot_precision_recall(hist, max_range=9)
    plot_fp_fn_counts(hist, hist_trk, max_range=9)
