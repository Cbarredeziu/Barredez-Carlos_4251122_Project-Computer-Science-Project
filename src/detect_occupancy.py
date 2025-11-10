# detect_occupancy.py
import cv2
import json
import os
import argparse
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from ultralytics import YOLO
from datetime import datetime

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def load_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["frame_size"]
    polys = {s["id"]: np.array(s["points"], dtype=np.int32) for s in data["spots"]}
    return (w, h), polys

def mask_from_polygon(poly_pts, shape_hw):
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    return mask

def mask_from_bbox(xyxy, shape_hw):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(x1,0); y1 = max(y1,0)
    h, w = shape_hw
    x2 = min(x2, w-1); y2 = min(y2, h-1)
    m = np.zeros(shape_hw, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m

def polygon_area(mask):
    return int((mask>0).sum())

def status_smoother_factory(history=5):
    return defaultdict(lambda: deque(maxlen=history))

def vote_status(deq):
    if not deq:
        return "free"
    ones = sum(1 for s in deq if s == "occupied")
    return "occupied" if ones > (len(deq) - ones) else "free"

def detect_occupancy(video_path, map_path, weights="yolov8n.pt",
                     conf=0.35, overlap_thr=0.15, history=5,
                     csv_out="test/occupancy_log.csv", json_out="test/occupancy_lastframe.json",
                     display=False, save_image=None):

    # Ensure output directories exist
    os.makedirs("test", exist_ok=True)
    
    (map_w, map_h), polys = load_map(map_path)

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

        for pid, spot_mask in spot_masks.items():
            best_overlap = 0.0
            best_det = None
            for d, dm in zip(dets, det_masks):
                inter_px = int((np.bitwise_and(spot_mask, dm) > 0).sum())
                frac = inter_px / max(1, spot_areas[pid])
                if frac > best_overlap:
                    best_overlap = frac
                    best_det = d
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

        iso = datetime.utcnow().isoformat() + "Z"
        for pid, info in per_spot_status.items():
            rows.append({
                "ts_utc": iso, "frame": frame_idx, "t_ms": t_ms,
                "spot_id": pid, "status": info["state"], "raw_status": info["raw"],
                "overlap": info["overlap"], "vehicle": info["vehicle"], 
                "vehicle_id": info["vehicle_id"], "det_conf": info["conf"]
            })

        # Always create visualization for saving
        vis = frame.copy()
        for pid, spot_mask in spot_masks.items():
            color = (0,180,0) if per_spot_status[pid]["state"] == "free" else (0,0,220)
            cnts, _ = cv2.findContours((spot_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 2)
            ys, xs = np.where(spot_mask>0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                status = per_spot_status[pid]["state"]
                txt = f'{pid}:{status}'
                cv2.putText(vis, txt, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        for d in dets:
            x1,y1,x2,y2 = map(int, d["xyxy"])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
            cv2.putText(vis, f'ID:{d["id"]} {d["cls"]} {d["conf"]:.2f}', (x1, max(15,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        occ = sum(1 for s in per_spot_status.values() if s["state"] == "occupied")
        free = len(per_spot_status) - occ
        cv2.rectangle(vis, (0,0), (260,60), (0,0,0), -1)
        cv2.putText(vis, f"Occupied: {occ}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Free: {free}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Always save visualization image to test directory
        if save_image:
            output_path = save_image
        else:
            # Default save path in test directory
            output_path = "test/parking_result.png"
        
        cv2.imwrite(output_path, vis)
        print(f"[OK] Visualization saved: {output_path}")

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

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"[OK] CSV saved: {csv_out}")

        last_frame = df[df["frame"] == df["frame"].max()]
        summary = {
            "ts_utc": last_frame["ts_utc"].iloc[-1],
            "frame": int(last_frame["frame"].iloc[-1]),
            "spots": {}
        }
        for pid in sorted(set(last_frame["spot_id"])):
            rec = last_frame[last_frame["spot_id"]==pid].iloc[-1]
            summary["spots"][pid] = {"status": rec["status"], "overlap": float(rec["overlap"])}
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] Last frame JSON: {json_out}")

def _cli():
    ap = argparse.ArgumentParser(description="Parking spot occupancy detection using mapping JSON")
    ap.add_argument("--video", required=True, help="Path to video")
    ap.add_argument("--map", required=True, help="Path to mapping JSON")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--overlap_thr", type=float, default=0.15)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--csv_out", default="test/occupancy_log.csv")
    ap.add_argument("--json_out", default="test/occupancy_lastframe.json")
    ap.add_argument("--save_image", default="test/parking_visualization.png", help="Path to save the visualization image")
    args = ap.parse_args()

    detect_occupancy(
        video_path=args.video, map_path=args.map, weights=args.weights,
        conf=args.conf, overlap_thr=args.overlap_thr, history=args.history,
        csv_out=args.csv_out, json_out=args.json_out, display=args.display,
        save_image=args.save_image
    )

if __name__ == "__main__":
    _cli()
