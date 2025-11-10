# main.py
import os
import argparse

from map_parking_zones import map_parking_zones
from detect_occupancy import detect_occupancy

def main():
    ap = argparse.ArgumentParser(description="Parking pipeline: mapping -> occupancy")
    ap.add_argument("--video", required=True, help="Path to base video (or image)")
    ap.add_argument("--map", default="test/parking_map.json", help="Path to mapping JSON")
    ap.add_argument("--snap_lines", action="store_true", help="(mapping) show detected reference lines")
    # Detection parameters
    ap.add_argument("--weights", default="yolov8x.pt", help="YOLO weights (e.g. yolov8n.pt or yolo11n.pt)")
    ap.add_argument("--conf", type=float, default=0.35, help="Minimum detection confidence")
    ap.add_argument("--overlap_thr", type=float, default=0.6, help="Minimum overlap (fraction) to mark as occupied")
    ap.add_argument("--history", type=int, default=5, help="Temporal smoothing window (frames)")
    ap.add_argument("--display", action="store_true", help="Show visualization in window")
    ap.add_argument("--csv_out", default="test/occupancy_log.csv", help="Output CSV")
    ap.add_argument("--json_out", default="test/occupancy_lastframe.json", help="Last frame state JSON")
    ap.add_argument("--save_image", default="test/parking_visualization.png", help="Path to save the visualization image")
    args = ap.parse_args()

    # 1) Check for parking map in test directory first, then specified path
    map_path = args.map
    if not os.path.exists(map_path):
        # Try looking in test directory if not found at specified path
        test_map_path = os.path.join("test", os.path.basename(map_path))
        if os.path.exists(test_map_path):
            map_path = test_map_path
            print(f"[INFO] Using existing map from test directory: {map_path}")
        else:
            print(f"[INFO] Map '{map_path}' does not exist. Opening mapping editor...")
            map_parking_zones(video_path=args.video, out_path=map_path, snap_lines=args.snap_lines)
            if not os.path.exists(map_path):
                print("[WARN] Mapping JSON was not saved. Exiting.")
                return

    # 2) With JSON ready, run occupancy detection
    print(f"[INFO] Using map '{map_path}'. Starting detection...")
    detect_occupancy(
        video_path=args.video,
        map_path=map_path,
        weights=args.weights,
        conf=args.conf,
        overlap_thr=args.overlap_thr,
        history=args.history,
        csv_out=args.csv_out,
        json_out=args.json_out,
        display=args.display,
        save_image=args.save_image
    )

if __name__ == "__main__":
    main()
