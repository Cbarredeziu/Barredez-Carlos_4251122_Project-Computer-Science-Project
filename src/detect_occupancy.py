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
    # COCO IDs used by YOLOv8/11 (may vary by model)
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def load_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["frame_size"]
    spots = data["spots"]
    polys = {}
    for s in spots:
        pts = np.array(s["points"], dtype=np.int32)
        polys[s["id"]] = pts
    return (w, h), polys, data

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

def iou_mask(a, b):
    inter = np.bitwise_and(a, b)
    union = np.bitwise_or(a, b)
    i = int((inter>0).sum())
    u = int((union>0).sum())
    return (i / u) if u > 0 else 0.0, i, u

def status_smoother_factory(history=5):
    # states per spot with sliding window
    return defaultdict(lambda: deque(maxlen=history))

def vote_status(deq):
    if not deq:
        return "free"
    ones = sum(1 for s in deq if s == "occupied")
    zeros = len(deq) - ones
    return "occupied" if ones > zeros else "free"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to video")
    ap.add_argument("--map", type=str, required=True, help="JSON with parking spots")
    ap.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (Ultralytics)")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--overlap_thr", type=float, default=0.15, help="Overlap fraction (spot) to mark as occupied")
    ap.add_argument("--history", type=int, default=5, help="History length for smoothing")
    ap.add_argument("--csv_out", type=str, default="occupancy_log.csv", help="Output CSV file")
    ap.add_argument("--json_out", type=str, default="occupancy_lastframe.json", help="Output JSON file (last frame)")
    ap.add_argument("--display", action="store_true", help="Show window with overlay")
    args = ap.parse_args()

    (map_w, map_h), polys, meta = load_map(args.map)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame of the video.")
    H, W = frame.shape[:2]
    if (W, H) != (map_w, map_h):
        print(f"[WARNING] Video size {W}x{H} differs from mapping {map_w}x{map_h}. Masks will be adapted to frame size.")

    # Precrear máscaras de plazas por tamaño del frame
    def build_spot_masks(h, w):
        masks = {}
        for pid, pts in polys.items():
            # escalar si el tamaño no coincide
            scale_x = w / map_w
            scale_y = h / map_h
            pts_scaled = np.int32(np.round(pts * np.array([scale_x, scale_y])))
            masks[pid] = mask_from_polygon(pts_scaled, (h,w))
        return masks

    spot_masks = build_spot_masks(H, W)
    spot_areas = {pid: polygon_area(m) for pid, m in spot_masks.items()}

    model = YOLO(args.weights)

    # logger a CSV
    rows = []
    smoother = status_smoother_factory(history=args.history)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if frame_idx == 1:
            H, W = frame.shape[:2]
            spot_masks = build_spot_masks(H, W)
            spot_areas = {pid: polygon_area(m) for pid, m in spot_masks.items()}

        # Inferencia YOLO
        results = model.predict(frame, conf=args.conf, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None: 
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                if cls_id in VEHICLE_CLASSES:
                    conf = float(b.conf.item())
                    x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
                    dets.append({"cls": VEHICLE_CLASSES[cls_id],
                                 "conf": conf,
                                 "xyxy": (x1,y1,x2,y2)})

        # Calcular ocupación por plaza
        per_spot_status = {}
        # Preconstruir máscaras de detección (por bbox)
        det_masks = []
        for d in dets:
            det_masks.append(mask_from_bbox(d["xyxy"], (H,W)))

        for pid, spot_mask in spot_masks.items():
            occupied = False
            best_overlap = 0.0
            best_det = None

            for d, dm in zip(dets, det_masks):
                inter = np.bitwise_and(spot_mask, dm)
                inter_px = int((inter>0).sum())
                frac = inter_px / max(1, spot_areas[pid])
                if frac > best_overlap:
                    best_overlap = frac
                    best_det = d
            if best_overlap >= args.overlap_thr:
                occupied = True

            # suavizado temporal (voto mayoritario)
            smoother[pid].append("occupied" if occupied else "free")
            state = vote_status(smoother[pid])
            per_spot_status[pid] = {
                "raw": "occupied" if occupied else "free",
                "state": state,
                "overlap": round(best_overlap, 3),
                "vehicle": best_det["cls"] if (best_det and occupied) else None,
                "conf": round(best_det["conf"], 3) if (best_det and occupied) else None
            }

        # Log CSV (una fila por plaza)
        iso = datetime.utcnow().isoformat() + "Z"
        for pid, info in per_spot_status.items():
            rows.append({
                "ts_utc": iso,
                "frame": frame_idx,
                "t_ms": t_ms,
                "spot_id": pid,
                "status": info["state"],        # suavizado
                "raw_status": info["raw"],      # instantáneo
                "overlap": info["overlap"],
                "vehicle": info["vehicle"],
                "det_conf": info["conf"]
            })

        # Visualización
        if args.display:
            vis = frame.copy()
            for pid, spot_mask in spot_masks.items():
                color = (0,180,0) if per_spot_status[pid]["state"] == "free" else (0,0,220)
                # pintar borde
                cnts, _ = cv2.findContours((spot_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, color, 2)
                # etiqueta
                # centroide rápido
                ys, xs = np.where(spot_mask>0)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    txt = f'{pid}:{per_spot_status[pid]["state"]}'
                    cv2.putText(vis, txt, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # dibuja bboxes de vehículos
            for d in dets:
                x1,y1,x2,y2 = map(int, d["xyxy"])
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
                cv2.putText(vis, f'{d["cls"]} {d["conf"]:.2f}', (x1, max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # HUD
            occ = sum(1 for s in per_spot_status.values() if s["state"] == "occupied")
            free = len(per_spot_status) - occ
            cv2.rectangle(vis, (0,0), (260,60), (0,0,0), -1)
            cv2.putText(vis, f"Occupied: {occ}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Free: {free}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("Parking Occupancy", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.csv_out, index=False, encoding="utf-8")
        print(f"[OK] CSV saved: {args.csv_out}")

        # JSON of the last frame (instantaneous state usable by another process)
        last_frame = df[df["frame"] == df["frame"].max()]
        summary = {
            "ts_utc": last_frame["ts_utc"].iloc[-1],
            "frame": int(last_frame["frame"].iloc[-1]),
            "spots": {}
        }
        for pid in sorted(set(last_frame["spot_id"])):
            rec = last_frame[last_frame["spot_id"]==pid].iloc[-1]
            summary["spots"][pid] = {
                "status": rec["status"],
                "overlap": float(rec["overlap"])
            }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON last frame: {args.json_out}")
