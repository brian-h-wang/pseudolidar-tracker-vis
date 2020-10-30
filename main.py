from tracker_vis.visualization import TrackingVisualizer
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trk_results", help="Path to the tracking results .txt file.")
    parser.add_argument("--pcd_dir", help="Path to a directory containing point cloud .bin files.")
    parser.add_argument("--image_dir", help="Path to a directory containing images for display.")
    parser.add_argument("--det", action='store_true', help="Use detections instead of tracking results")
    parser.add_argument("--n_skip", default=1, type=int, help="How many frames to skip")
    parser.add_argument("--show_images", default=False, action="store_true", help="Show images alongside point cloud")
    args = parser.parse_args()

    results_path = Path(args.trk_results)
    pcd_dir = Path(args.pcd_dir)
    image_dir = Path(args.image_dir)

    vis = TrackingVisualizer(results_path=results_path, pointcloud_path=pcd_dir, image_path=image_dir, fps=60,
                             load_detections=args.det, n_skip=args.n_skip, show_images=args.show_images)
    try:
        vis.visualize_all()
    finally:
        vis.plot_ranges()