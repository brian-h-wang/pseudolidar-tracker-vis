from analysis.detections_analysis import DetectionsHistogram, plot_means_stddevs, plot_precision_recall
from tracker_vis.tracking import TrackerResults

if __name__ == "__main__":
    det = TrackerResults.load_from_detections("data/detections_epoch70")
    gt = TrackerResults.load("data/tracking_gt_updated.txt")
    trk = TrackerResults.load("data/trk_highest_R.txt")

    hist = DetectionsHistogram(det, gt, assoc_threshold=1.0)
    hist_trk = DetectionsHistogram(trk, gt, assoc_threshold=1.0)
    print("FP")
    print(hist.false_positives_count)
    print(hist_trk.false_positives_count)

    print("FN")
    print(hist.false_negatives_count)
    print(hist_trk.false_negatives_count)

    plot_means_stddevs(hist, hist_trk, max_range=9)
    # plot_precision_recall(hist, max_range=9)
