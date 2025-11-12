# main.py
import os
import argparse

from map_parking_zones import map_parking_zones
from vehicle_detector import detect_occupancy
from edit_parking_zones import EnhancedParkingZoneEditor

def main():
    ap = argparse.ArgumentParser(description="Parking pipeline: automatic processing of folder contents")
    ap.add_argument("--map", default="test/parking_map.json", help="Path to mapping JSON")
    ap.add_argument("--snap_lines", action="store_true", help="(mapping) show detected reference lines")
    # Editor mode
    ap.add_argument("--edit", action="store_true", help="Launch parking zone editor instead of detection")
    ap.add_argument("--image", help="Image file for parking zone editor (required with --edit)")
    # Detection parameters
    ap.add_argument("--weights", default="yolov8x.pt", help="YOLO weights (e.g. yolov8n.pt or yolo11n.pt)")
    ap.add_argument("--conf", type=float, default=0.35, help="Minimum detection confidence")
    ap.add_argument("--overlap_thr", type=float, default=0.7, help="Minimum overlap (fraction) to mark as occupied")
    ap.add_argument("--history", type=int, default=5, help="Temporal smoothing window (frames)")
    ap.add_argument("--display", action="store_true", help="Show visualization in window")
    ap.add_argument("--no_save_image", action="store_true", help="Disable saving visualization images")
    ap.add_argument("--no_blur", action="store_true", help="Disable vehicle blurring (blur is enabled by default for privacy)")
    args = ap.parse_args()
    
    # Always save images by default, unless --no_save_image is specified
    args.save_image = not args.no_save_image
    # Always blur vehicles by default for privacy, unless --no_blur is specified
    args.blur_vehicles = not args.no_blur

    # Check if edit mode is requested
    if args.edit:
        # Launch parking zone editor
        if not args.image:
            print("[ERROR] --image argument is required when using --edit mode")
            print("Example: python main.py --edit --image ../data/inputs/parkingplace_cars1.png")
            return
        
        if not os.path.exists(args.image):
            print(f"[ERROR] Image file not found: {args.image}")
            return
        
        # Detect path for editor mode  
        if os.path.exists("test"):
            test_base = "test"
        elif os.path.exists("../test"):
            test_base = "../test"
        elif os.path.exists("src/test"):
            test_base = "test"
        else:
            test_base = "../test" if os.path.basename(os.getcwd()) == "src" else "test"
        
        map_path = os.path.join(test_base, "parking_grind", "parking_map.json")
        
        print("Launching Enhanced Parking Zone Editor...")
        print("=========================================")
        try:
            editor = EnhancedParkingZoneEditor(args.image, map_path if os.path.exists(map_path) else None)
            editor.map_path = map_path  # Set output path
            editor.run()
            
            # Final save prompt only if user hasn't saved during the session
            if editor.zones and not editor.has_saved:
                print(f"\nFinal save to {map_path}?")
                response = input("Save changes? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    editor.save_map(map_path)
                    print(f"Parking zones saved to: {map_path}")
                else:
                    print("Changes not saved")
            elif editor.has_saved:
                print(f"\n[INFO] Zones already saved during editing session.")
        except Exception as e:
            print(f"[ERROR] Editor failed: {e}")
        
        return  # Exit after editor, don't run detection

    # 1) Check for parking map and create if needed
    # Detect if running from src directory or root directory
    if os.path.exists("data") and os.path.exists("test"):
        # Running from src directory
        data_base = "data"
        test_base = "test"
    elif os.path.exists("../data") and os.path.exists("../test"):
        # Running from src directory with relative paths
        data_base = "../data"
        test_base = "../test"
    elif os.path.exists("src/test") and os.path.exists("data"):
        # Running from root directory
        data_base = "data"
        test_base = "test"
    else:
        # Default paths
        data_base = "../data" if os.path.basename(os.getcwd()) == "src" else "data"
        test_base = "../test" if os.path.basename(os.getcwd()) == "src" else "test"
    
    # Set up paths based on detected location
    map_path = os.path.join(test_base, "parking_grind", "parking_map.json")
    inputs_dir = os.path.join(data_base, "inputs")
    grid_dir = os.path.join(data_base, "grid_photos")
    parking_grind_dir = os.path.join(test_base, "parking_grind")
    results_dir = os.path.join(test_base, "results")
    
    # Create necessary directories
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(parking_grind_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "grid_results"), exist_ok=True)
    
    if not os.path.exists(map_path):
        # Try looking in old test directory for backward compatibility
        old_map_path = os.path.join(test_base, "parking_map.json")
        if os.path.exists(old_map_path):
            print(f"[INFO] Found existing map at old location: {old_map_path}")
            print(f"[INFO] Moving to new location: {map_path}")
            # Move the old file to new location
            import shutil
            shutil.move(old_map_path, map_path)
            # Also move visualization if it exists
            old_vis_path = os.path.join(test_base, "parking_map_zones_visualization.png")
            if os.path.exists(old_vis_path):
                new_vis_path = os.path.join(parking_grind_dir, "parking_map_zones_visualization.png")
                shutil.move(old_vis_path, new_vis_path)
        else:
            # Look for any image in grid_photos to use for mapping
            mapping_image = None
            if os.path.exists(grid_dir):
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    grid_files = [f for f in os.listdir(grid_dir) if f.lower().endswith(ext)]
                    if grid_files:
                        mapping_image = os.path.join(grid_dir, grid_files[0])
                        break
            
            if mapping_image:
                print(f"[INFO] No parking map found. Using '{mapping_image}' for zone mapping...")
                map_parking_zones(video_path=mapping_image, out_path=map_path, snap_lines=args.snap_lines)
                if not os.path.exists(map_path):
                    print("[WARN] Mapping JSON was not saved. Exiting.")
                    return
            else:
                print(f"[ERROR] No parking map found and no images in '{grid_dir}' for mapping.")
                print(f"[INFO] Please either:")
                print(f"  1. Place a reference image in '{grid_dir}' folder, or")
                print(f"  2. Create a parking map manually")
                return

    # 2) With JSON ready, look for detection input files
    
    # Check for files in inputs directory (cars images/videos)
    input_files = []
    if os.path.exists(inputs_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.mp4', '.avi', '.mov', '.mkv']:
            input_files.extend([f for f in os.listdir(inputs_dir) if f.lower().endswith(ext)])
    
    # Check for files in grid_photos directory (reference/mapping images)
    grid_files = []
    if os.path.exists(grid_dir):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            grid_files.extend([f for f in os.listdir(grid_dir) if f.lower().endswith(ext)])
    
    # Process regular input files (cars images/videos)
    if input_files:
        print(f"[INFO] Found {len(input_files)} input file(s) in '{inputs_dir}' for detection.")
        
        # Use consolidated output files for all inputs
        csv_out = os.path.join(results_dir, "data", "occupancy_log.csv")
        json_out = os.path.join(results_dir, "data", "occupancy_summary.json")
        
        for input_file in input_files:
            input_path = os.path.join(inputs_dir, input_file)
            print(f"[INFO] Processing input: {input_file}")
            
            detect_occupancy(
                video_path=input_path,
                map_path=map_path,
                weights=args.weights,
                conf=args.conf,
                overlap_thr=args.overlap_thr,
                history=args.history,
                csv_out=csv_out,
                json_out=json_out,
                display=args.display,
                save_image=args.save_image,
                results_dir=results_dir,
                blur_vehicles=args.blur_vehicles
            )
    
    # Process grid photos (reference/mapping images)
    if grid_files:
        print(f"[INFO] Found {len(grid_files)} grid photo(s) in '{grid_dir}' for reference analysis.")
        
        # Use consolidated output files for all grid photos
        csv_out = os.path.join(results_dir, "data", "grid_analysis_log.csv")
        json_out = os.path.join(results_dir, "data", "grid_analysis_summary.json")
        
        for grid_file in grid_files:
            grid_path = os.path.join(grid_dir, grid_file)
            print(f"[INFO] Processing grid photo: {grid_file}")
            
            detect_occupancy(
                video_path=grid_path,
                map_path=map_path,
                weights=args.weights,
                conf=args.conf,
                overlap_thr=args.overlap_thr,
                history=args.history,
                csv_out=csv_out,
                json_out=json_out,
                display=args.display,
                save_image=args.save_image,
                results_dir=results_dir,
                blur_vehicles=args.blur_vehicles
            )
    
    # Show information if no files found
    if not input_files and not grid_files:
        print(f"[INFO] No files found for processing.")
        print(f"[INFO] Place images/videos with cars in '{inputs_dir}' folder.")
        print(f"[INFO] Place reference grid photos in '{grid_dir}' folder.")
        print(f"[INFO] Skipping detection analysis.")

if __name__ == "__main__":
    main()
