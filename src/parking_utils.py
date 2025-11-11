# parking_utils.py
# Utility functions for parking zone management and geometric calculations

import cv2
import json
import os
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def load_map(json_path):
    """Load parking zone map from JSON file with support for multiple zone types"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["frame_size"]
    
    # New format with zone types
    if "zones" in data:
        # Load all zones with their types
        all_zones = {z["id"]: {
            "points": np.array(z["points"], dtype=np.int32),
            "type": z.get("type", "P"),
            "zone_name": z.get("zone_name", "parking")
        } for z in data["zones"]}
        
        # Filter parking zones for occupancy detection (exclude traffic zones)
        parking_zones = {zid: zone["points"] for zid, zone in all_zones.items() 
                        if zone["type"] in ["P", "D"]}  # Include parking and disabled zones
        
        return (w, h), parking_zones, all_zones
    else:
        # Backward compatibility with old format
        polys = {s["id"]: np.array(s["points"], dtype=np.int32) for s in data["spots"]}
        # Convert to new format for consistency
        all_zones = {sid: {"points": points, "type": "P", "zone_name": "parking"} 
                    for sid, points in polys.items()}
        return (w, h), polys, all_zones

def mask_from_polygon(poly_pts, shape_hw):
    """Create a binary mask from polygon points"""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)
    return mask

def mask_from_bbox(xyxy, shape_hw):
    """Create a binary mask from bounding box coordinates"""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(x1,0); y1 = max(y1,0)
    h, w = shape_hw
    x2 = min(x2, w-1); y2 = min(y2, h-1)
    m = np.zeros(shape_hw, dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m

def polygon_area(mask):
    """Calculate area of a polygon from its mask"""
    return int((mask>0).sum())

def status_smoother_factory(history=5):
    """Factory function to create status smoothing deques for each parking spot"""
    return defaultdict(lambda: deque(maxlen=history))

def vote_status(deq):
    """Vote on the most common status in a deque"""
    if not deq:
        return "free"
    ones = sum(1 for s in deq if s == "occupied")
    return "occupied" if ones > (len(deq) - ones) else "free"

def get_unique_filename(base_path, extension=".png"):
    """Generate unique filename by adding incremental numbers if file exists"""
    if not os.path.exists(base_path + extension):
        return base_path + extension
    
    counter = 1
    while os.path.exists(f"{base_path}_{counter}{extension}"):
        counter += 1
    return f"{base_path}_{counter}{extension}"