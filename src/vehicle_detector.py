# vehicle_detector.py
# Main vehicle detection and parking occupancy analysis

import cv2
import os
import argparse
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime

# Import utility functions from parking_utils
from parking_utils import (
    VEHICLE_CLASSES, load_map, mask_from_polygon, mask_from_bbox, polygon_area,
    status_smoother_factory, vote_status, get_unique_filename
)

def detect_occupancy(video_path, map_path, weights="yolov8n.pt",
                     conf=0.35, overlap_thr=0.15, history=5,
                     csv_out="../test/occupancy_log.csv", json_out="../test/occupancy_lastframe.json",
                     display=False, save_image=True, results_dir="../test/results"):

    # Ensure output directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "grid_results"), exist_ok=True)
    
    (map_w, map_h), polys, all_zones = load_map(map_path)

    # Check if input is an image file
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    is_image = any(video_path.lower().endswith(ext) for ext in image_extensions)
    
    if is_image:
        # Handle image input
        frame = cv2.imread(video_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {video_path}")
        cap = None
        ret = True
    else:
        # Handle video input
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read the first frame.")
    H, W = frame.shape[:2]
    if (W, H) != (map_w, map_h):
        print(f"[WARNING] Video {W}x{H} differs from mapping {map_w}x{map_h}. Rescaling masks...")

    def build_spot_masks(h, w):
        masks = {}
        for pid, pts in polys.items():
            sx = w / map_w
            sy = h / map_h
            pts_scaled = np.int32(np.round(pts * np.array([sx, sy])))
            masks[pid] = mask_from_polygon(pts_scaled, (h, w))
        return masks

    spot_masks = build_spot_masks(H, W)
    spot_areas = {pid: polygon_area(m) for pid, m in spot_masks.items()}

    model = YOLO(weights)
    rows = []
    smoother = status_smoother_factory(history=history)
    vehicle_id_counter = 0  # Counter for assigning unique vehicle IDs

    frame_idx = 0
    while True:
        if is_image:
            # For images, we already have the frame and only process once
            if frame_idx > 0:
                break
            t_ms = 0
        else:
            # For videos, read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
        frame_idx += 1
        if frame_idx == 1:
            H, W = frame.shape[:2]
            spot_masks = build_spot_masks(H, W)
            spot_areas = {pid: polygon_area(m) for pid, m in spot_masks.items()}

        results = model.predict(frame, conf=conf, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                if cls_id in VEHICLE_CLASSES:
                    vehicle_id_counter += 1  # Increment counter for each detected vehicle
                    dets.append({
                        "id": vehicle_id_counter,
                        "cls": VEHICLE_CLASSES[cls_id],
                        "conf": float(b.conf.item()),
                        "xyxy": tuple(float(v) for v in b.xyxy[0].tolist())
                    })

        per_spot_status = {}
        det_masks = [mask_from_bbox(d["xyxy"], (H, W)) for d in dets]
        
        # Track which vehicles are assigned to parking spots
        assigned_vehicles = set()

        for pid, spot_mask in spot_masks.items():
            best_overlap = 0.0
            best_det = None
            best_det_idx = -1
            for i, (d, dm) in enumerate(zip(dets, det_masks)):
                inter_px = int((np.bitwise_and(spot_mask, dm) > 0).sum())
                frac = inter_px / max(1, spot_areas[pid])
                if frac > best_overlap:
                    best_overlap = frac
                    best_det = d
                    best_det_idx = i
            occupied = best_overlap >= overlap_thr
            smoother[pid].append("occupied" if occupied else "free")
            per_spot_status[pid] = {
                "raw": "occupied" if occupied else "free",
                "state": vote_status(smoother[pid]),
                "overlap": round(best_overlap, 3),
                "vehicle": best_det["cls"] if (best_det and occupied) else None,
                "vehicle_id": best_det["id"] if (best_det and occupied) else None,
                "conf": round(best_det["conf"], 3) if (best_det and occupied) else None
            }
            
            # Track assigned vehicle
            if occupied and best_det_idx >= 0:
                assigned_vehicles.add(best_det_idx)
        
        # Find vehicles not in any parking space
        unassigned_vehicles = []
        for i, det in enumerate(dets):
            if i not in assigned_vehicles:
                # Check overlaps with all zone types
                max_parking_overlap = 0.0
                in_traffic_zone = False
                in_no_parking_zone = False
                zone_status = "illegally_parked"
                
                dm = det_masks[i]
                vehicle_area = int((dm > 0).sum())
                
                if vehicle_area > 0:
                    for zone_id, zone_info in all_zones.items():
                        zone_mask = mask_from_polygon(zone_info["points"], (H, W))
                        inter_px = int((np.bitwise_and(zone_mask, dm) > 0).sum())
                        frac = inter_px / vehicle_area
                        
                        if zone_info["type"] in ["P", "D"]:  # Parking or disabled zones
                            max_parking_overlap = max(max_parking_overlap, frac)
                        elif zone_info["type"] == "T" and frac > 0.3:  # Traffic zone
                            in_traffic_zone = True
                        elif zone_info["type"] == "N" and frac > 0.3:  # No-parking zone
                            in_no_parking_zone = True
                
                # Determine vehicle status based on zone overlaps
                if in_traffic_zone:
                    zone_status = "in_traffic_zone"
                elif in_no_parking_zone:
                    zone_status = "in_no_parking_zone"
                elif max_parking_overlap > 0.1:
                    zone_status = "partially_in_parking_zone"
                else:
                    zone_status = "illegally_parked"
                
                unassigned_vehicles.append({
                    "vehicle_id": det["id"],
                    "vehicle_type": det["cls"],
                    "confidence": round(det["conf"], 3),
                    "bbox": det["xyxy"],
                    "max_parking_overlap": round(max_parking_overlap, 3),
                    "status": zone_status,
                    "in_traffic_zone": in_traffic_zone,
                    "in_no_parking_zone": in_no_parking_zone
                })

        iso = datetime.utcnow().isoformat() + "Z"
        # Get source filename for identification
        source_filename = os.path.basename(video_path)
        
        for pid, info in per_spot_status.items():
            rows.append({
                "source_file": source_filename,
                "ts_utc": iso, "frame": frame_idx, "t_ms": t_ms,
                "spot_id": pid, "status": info["state"], "raw_status": info["raw"],
                "overlap": info["overlap"], "vehicle": info["vehicle"], 
                "vehicle_id": info["vehicle_id"], "det_conf": info["conf"],
                "type": "parking_spot"
            })
        
        # Add unassigned vehicles to the log
        for vehicle in unassigned_vehicles:
            rows.append({
                "source_file": source_filename,
                "ts_utc": iso, "frame": frame_idx, "t_ms": t_ms,
                "spot_id": f"UNASSIGNED", "status": vehicle["status"], "raw_status": vehicle["status"],
                "overlap": vehicle["max_parking_overlap"], "vehicle": vehicle["vehicle_type"],
                "vehicle_id": vehicle["vehicle_id"], "det_conf": vehicle["confidence"],
                "type": "unassigned_vehicle"
            })

        # Create visualization
        vis = create_visualization(frame, all_zones, per_spot_status, dets, 
                                 assigned_vehicles, unassigned_vehicles, video_path, H, W)

        # Always save visualization image
        if save_image:
            save_visualization_image(vis, video_path, results_dir)

        # Show display window if requested
        if display:
            cv2.imshow("Parking Occupancy", vis)
            if is_image:
                # For images, wait indefinitely for user to press any key
                print("Press any key to close the display window...")
                cv2.waitKey(0)
                break
            else:
                # For videos, wait 1ms and check for quit keys
                if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                    break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # Save CSV and JSON data
    save_detection_results(rows, csv_out, json_out, video_path)
    
    return per_spot_status

def create_visualization(frame, all_zones, per_spot_status, dets, assigned_vehicles, 
                        unassigned_vehicles, video_path, H, W):
    """Create visualization with zones and vehicle detection results"""
    vis = frame.copy()
    is_grid_analysis = "grid_photos" in video_path
    
    # Define zone type colors
    zone_colors = {
        "P": {"free": (0,180,0), "occupied": (0,0,220)},      # Green/Red for parking
        "T": {"free": (0,165,255), "occupied": (0,100,255)},  # Orange for traffic
        "N": {"free": (0,0,255), "occupied": (0,0,180)},      # Red for no-parking
        "D": {"free": (255,0,0), "occupied": (200,0,0)}       # Blue for disabled (accessibility standard)
    }
    
    # Draw all zones (including non-parking zones for reference)
    for zone_id, zone_info in all_zones.items():
        zone_type = zone_info["type"]
        zone_points = zone_info["points"]
        
        # Create mask for this zone
        zone_mask = mask_from_polygon(zone_points, (H, W))
        
        if zone_id in per_spot_status:
            # This is a parking/disabled zone with occupancy status
            status = per_spot_status[zone_id]["state"]
            color = zone_colors[zone_type][status]
        else:
            # This is a traffic/no-parking zone (reference only)
            color = zone_colors[zone_type]["free"]
        
        cnts, _ = cv2.findContours((zone_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        
        ys, xs = np.where(zone_mask>0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            if zone_id in per_spot_status:
                status = per_spot_status[zone_id]["state"]
                txt = f'{zone_id}:{status}'
            else:
                zone_name = zone_info["zone_name"]
                txt = f'{zone_id}:{zone_name}'
            cv2.putText(vis, txt, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Create a lookup for unassigned vehicle status
    unassigned_status = {v["vehicle_id"]: v["status"] for v in unassigned_vehicles}
    
    # Draw all detected vehicles with status-based colors
    for i, d in enumerate(dets):
        x1,y1,x2,y2 = map(int, d["xyxy"])
        
        if i in assigned_vehicles:
            color = (255, 255, 255)  # White for properly parked
            label = f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f} [PARKED]'
        else:
            # Color based on unassigned vehicle status
            status = unassigned_status.get(d["id"], "illegally_parked")
            if status == "in_traffic_zone":
                color = (255, 165, 0)  # Orange for traffic
                label = f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f} [TRAFFIC]'
            elif status == "in_no_parking_zone":
                color = (0, 0, 255)  # Red for no-parking violation
                label = f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f} [NO-PARK]'
            elif status == "partially_in_parking_zone":
                color = (255, 255, 0)  # Yellow for partial parking
                label = f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f} [PARTIAL]'
            else:
                color = (0, 0, 255)  # Red for illegal
                label = f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f} [ILLEGAL]'
        
        # Draw rectangle (always)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        
        # Draw label only for regular detection (not grid analysis)
        if not is_grid_analysis:
            cv2.putText(vis, label, (x1, max(15,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Add information overlay
    add_info_overlay(vis, all_zones, per_spot_status, assigned_vehicles, 
                    unassigned_vehicles, is_grid_analysis)
    
    return vis

def add_info_overlay(vis, all_zones, per_spot_status, assigned_vehicles, 
                    unassigned_vehicles, is_grid_analysis):
    """Add information overlay to visualization"""
    # Count zones by type
    zone_counts = {"P": 0, "T": 0, "N": 0, "D": 0}
    for zone_info in all_zones.values():
        zone_type = zone_info["type"]
        if zone_type in zone_counts:
            zone_counts[zone_type] += 1
    
    if is_grid_analysis:
        # Grid analysis: show only zone counts
        cv2.rectangle(vis, (0,0), (280,100), (0,0,0), -1)
        cv2.putText(vis, f"Parking zones: {zone_counts['P']}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Traffic zones: {zone_counts['T']}", (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"No-parking zones: {zone_counts['N']}", (8,56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Disabled zones: {zone_counts['D']}", (8,74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    else:
        # Regular detection: show vehicle status
        occ = sum(1 for s in per_spot_status.values() if s["state"] == "occupied")
        free = len(per_spot_status) - occ
        
        traffic_vehicles = sum(1 for v in unassigned_vehicles if v["status"] == "in_traffic_zone")
        illegal_vehicles = sum(1 for v in unassigned_vehicles if v["status"] in ["illegally_parked", "in_no_parking_zone"])
        partial_vehicles = sum(1 for v in unassigned_vehicles if v["status"] == "partially_in_parking_zone")
        
        cv2.rectangle(vis, (0,0), (320,120), (0,0,0), -1)
        cv2.putText(vis, f"Occupied: {occ}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Free: {free}", (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Traffic: {traffic_vehicles}", (8,56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Partial: {partial_vehicles}", (8,74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Illegal: {illegal_vehicles}", (8,92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

def save_visualization_image(vis, video_path, results_dir="../test/results"):
    """Save visualization image with appropriate filename"""
    input_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    if "grid_photos" in video_path:
        # Grid analysis: always use the same filename (overwrite previous)
        output_path = os.path.join(results_dir, "grid_results", f"{input_basename}_analysis.png")
    else:
        # Regular detection: use unique filenames
        base_name = os.path.join(results_dir, "images", f"{input_basename}_result")
        output_path = get_unique_filename(base_name, ".png")
    
    cv2.imwrite(output_path, vis)
    print(f"[OK] Visualization saved: {output_path}")

def save_detection_results(rows, csv_out, json_out, video_path):
    """Save detection results to CSV and JSON files"""
    if rows:
        df = pd.DataFrame(rows)
        
        # Append to existing CSV or create new one
        if os.path.exists(csv_out):
            # Append to existing file
            df.to_csv(csv_out, mode='a', header=False, index=False, encoding="utf-8")
            print(f"[OK] CSV appended: {csv_out}")
        else:
            # Create new file with headers
            df.to_csv(csv_out, index=False, encoding="utf-8")
            print(f"[OK] CSV created: {csv_out}")

        # Update consolidated JSON summary
        source_filename = os.path.basename(video_path)
        last_frame = df[df["frame"] == df["frame"].max()]
        
        # Load existing JSON or create new structure
        consolidated_summary = {}
        if os.path.exists(json_out):
            try:
                with open(json_out, "r", encoding="utf-8") as f:
                    consolidated_summary = json.load(f)
            except:
                consolidated_summary = {}
        
        # Add this source file's results
        file_summary = {
            "ts_utc": last_frame["ts_utc"].iloc[-1],
            "frame": int(last_frame["frame"].iloc[-1]),
            "spots": {},
            "unassigned_vehicles": []
        }
        
        # Add parking spots info
        parking_spots = last_frame[last_frame["type"] == "parking_spot"]
        for pid in sorted(set(parking_spots["spot_id"])):
            rec = parking_spots[parking_spots["spot_id"]==pid].iloc[-1]
            file_summary["spots"][pid] = {"status": rec["status"], "overlap": float(rec["overlap"])}
        
        # Add unassigned vehicles info
        unassigned = last_frame[last_frame["type"] == "unassigned_vehicle"]
        for _, rec in unassigned.iterrows():
            file_summary["unassigned_vehicles"].append({
                "vehicle_id": rec["vehicle_id"],
                "vehicle_type": rec["vehicle"],
                "status": rec["status"],
                "confidence": float(rec["det_conf"]),
                "max_zone_overlap": float(rec["overlap"])
            })
        
        consolidated_summary[source_filename] = file_summary
        
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(consolidated_summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON updated: {json_out}")

def _cli():
    """Command line interface"""
    ap = argparse.ArgumentParser(description="Parking spot occupancy detection using mapping JSON")
    ap.add_argument("--video", required=True, help="Path to video")
    ap.add_argument("--map", required=True, help="Path to mapping JSON")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--overlap_thr", type=float, default=0.15)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--csv_out", default="../test/occupancy_log.csv")
    ap.add_argument("--json_out", default="../test/occupancy_lastframe.json")
    ap.add_argument("--save_image", default=True, help="Save visualization image (default: True)")
    args = ap.parse_args()

    detect_occupancy(
        video_path=args.video, map_path=args.map, weights=args.weights,
        conf=args.conf, overlap_thr=args.overlap_thr, history=args.history,
        csv_out=args.csv_out, json_out=args.json_out, display=args.display,
        save_image=args.save_image
    )

if __name__ == "__main__":
    _cli()