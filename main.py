from tracker_vis.visualization import TrackingVisualizer
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trk_results", help="Path to the tracking results .txt file.")
    parser.add_argument("--pcd_dir", help="Path to a directory containing point cloud .bin files.")
    parser.add_argument("--det", action='store_true', help="Use detections instead of tracking results")
    args = parser.parse_args()

    results_path = Path(args.trk_results)
    pcd_dir = Path(args.pcd_dir)

    vis = TrackingVisualizer(results_path=results_path, pointcloud_path=pcd_dir, fps=60,
                             load_detections=args.det)
    try:
        vis.visualize_all()
    finally:
        vis.plot_ranges()