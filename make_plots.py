from analysis.detections_analysis import *
from tracker_vis.tracking import TrackerResults

if __name__ == "__main__":
    det = TrackerResults.load_from_detections("data/detections_epoch70")
    base = TrackerResults.load_from_detections("data/dbscan_baseline")
    gt = TrackerResults.load("data/tracking_gt_updated.txt")
    trk = TrackerResults.load("data/tracker_results_final.txt")

    # trk_var = TrackerResults.load_with_variance("data/trk_highest_R.txt", "data/variances.txt")

    binsize=0.5
    hist = DetectionsHistogram(det, gt, assoc_threshold=1.0, hist_bin_size=binsize)
    hist_trk = DetectionsHistogram(trk, gt, assoc_threshold=1.0, hist_bin_size=binsize)
    hist_base = DetectionsHistogram(base, gt, assoc_threshold=1.0, hist_bin_size=binsize)

    # plot_means_stddevs(hist, hist_trk, max_range=9)


    # plot_precision_recall(hist, hist_trk, hist_base, max_range=8)
    plot_fp_fn_counts(hist, hist_trk, hist_base, max_range=9)
