# map_parking_zones.py
import cv2
import json
import os
import argparse
import numpy as np
from datetime import datetime

def get_unique_filename(base_path, extension=".png"):
    """Generate unique filename by adding incremental numbers if file exists"""
    if not os.path.exists(base_path + extension):
        return base_path + extension
    
    counter = 1
    while os.path.exists(f"{base_path}_{counter}{extension}"):
        counter += 1
    return f"{base_path}_{counter}{extension}"

def detect_lines(img, canny1=80, canny2=160, hough_thresh=80, min_len=60, max_gap=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 1.2)
    edges = cv2.Canny(gray, canny1, canny2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
    return edges, lines

class PolygonEditor:
    def __init__(self, frame):
        self.base = frame.copy()
        self.show = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.polygons = []  # list of {"id": str, "type": str, "points": [(x,y),...]}
        self.curr_pts = []
        self.next_id = 1
        self.zone_types = {
            "P": {"name": "parking", "color": (0, 255, 0), "prefix": "P"},      # Green for parking
            "T": {"name": "traffic", "color": (0, 165, 255), "prefix": "T"},    # Orange for traffic  
            "N": {"name": "no_parking", "color": (0, 0, 255), "prefix": "N"},   # Red for no-parking
            "D": {"name": "disabled", "color": (255, 0, 0), "prefix": "D"}      # Blue for disabled (accessibility standard)
        }
        self.current_zone_type = "P"  # Default to parking zones
        self.show_help = False  # Toggle for showing full menu

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.curr_pts.append((x,y))
            self.redraw()

    def redraw(self):
        self.show = self.base.copy()
        # saved polygons with different colors by zone type
        for poly in self.polygons:
            pts = np.array(poly["points"], dtype=np.int32)
            zone_type = poly.get("type", "P")
            color = self.zone_types[zone_type]["color"]
            cv2.polylines(self.show, [pts], isClosed=True, thickness=2, color=color)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(self.show, poly["id"], c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # polygon being edited
        if self.curr_pts:
            pts = np.array(self.curr_pts, dtype=np.int32)
            current_color = self.zone_types[self.current_zone_type]["color"]
            cv2.polylines(self.show, [pts], isClosed=False, thickness=2, color=current_color)
            for p in self.curr_pts:
                cv2.circle(self.show, p, 4, current_color, -1)

        # Minimal HUD - only show current zone type and help hint
        if self.show_help:
            # Show full menu
            hud_height = 80
            cv2.rectangle(self.show, (0,0), (self.w, hud_height), (0,0,0), -1)
            
            # Main controls
            hud1 = "[LMB] points  [ENTER] close  [U] undo  [N] new  [S] save JSON  [Q] quit  [H] hide menu"
            cv2.putText(self.show, hud1, (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            # Zone type controls
            hud2 = "Zone Types: [1] Parking  [2] Traffic  [3] No-Parking  [4] Disabled"
            cv2.putText(self.show, hud2, (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
            # Current zone type indicator
            current_zone_name = self.zone_types[self.current_zone_type]["name"]
            current_color = self.zone_types[self.current_zone_type]["color"]
            hud3 = f"Current Zone Type: {current_zone_name.upper()}"
            cv2.putText(self.show, hud3, (8,58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2, cv2.LINE_AA)
        else:
            # Minimal overlay - only current zone type at top-right corner
            current_zone_name = self.zone_types[self.current_zone_type]["name"]
            current_color = self.zone_types[self.current_zone_type]["color"]
            text = f"Zone: {current_zone_name.upper()} [H for help]"
            
            # Get text size to create background box
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Position at top-right corner
            x_pos = self.w - text_width - 15
            y_pos = 25
            
            # Semi-transparent background
            overlay = self.show.copy()
            cv2.rectangle(overlay, (x_pos - 5, y_pos - text_height - 5), 
                         (x_pos + text_width + 5, y_pos + baseline + 5), (0, 0, 0), -1)
            self.show = cv2.addWeighted(overlay, 0.6, self.show, 0.4, 0)
            
            # Draw text
            cv2.putText(self.show, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2, cv2.LINE_AA)

    def close_current(self):
        if len(self.curr_pts) >= 3:
            prefix = self.zone_types[self.current_zone_type]["prefix"]
            pid = f"{prefix}{self.next_id}"
            zone_name = self.zone_types[self.current_zone_type]["name"]
            self.polygons.append({
                "id": pid, 
                "type": self.current_zone_type,
                "zone_name": zone_name,
                "points": self.curr_pts.copy()
            })
            self.next_id += 1
        self.curr_pts = []
        self.redraw()

def map_parking_zones(video_path, out_path="test/parking_grind/parking_map.json", snap_lines=False):
    # Load first frame (or image)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Invalid path: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video/image.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read the first frame.")

    # Ensure parking_grind directory exists
    os.makedirs("test/parking_grind", exist_ok=True)
    
    editor = PolygonEditor(frame)

    # Line detection (visual reference)
    _, lines = detect_lines(frame)
    overlay = editor.base.copy()
    if lines is not None:
        for (x1,y1,x2,y2) in lines[:,0,:]:
            cv2.line(overlay, (x1,y1), (x2,y2), (255,120,0), 2)
    if snap_lines:
        editor.base = cv2.addWeighted(overlay, 0.35, editor.base, 0.65, 0)
    editor.redraw()

    win = "Parking Mapper"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, editor.mouse)

    while True:
        cv2.imshow(win, editor.show)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):      # ENTER
            editor.close_current()
        elif key in (ord('n'), ord('N')):
            editor.curr_pts = []
            editor.redraw()
        elif key in (ord('u'), ord('U')):
            if editor.curr_pts:
                editor.curr_pts.pop()
            elif editor.polygons:
                editor.polygons.pop()
                editor.next_id = max(1, len(editor.polygons)+1)
            editor.redraw()
        elif key in (ord('h'), ord('H')):  # Toggle help menu
            editor.show_help = not editor.show_help
            editor.redraw()
        elif key == ord('1'):  # Parking zones
            editor.current_zone_type = "P"
            editor.redraw()
        elif key == ord('2'):  # Traffic zones
            editor.current_zone_type = "T"
            editor.redraw()
        elif key == ord('3'):  # No-parking zones
            editor.current_zone_type = "N"
            editor.redraw()
        elif key == ord('4'):  # Disabled parking zones
            editor.current_zone_type = "D"
            editor.redraw()
        elif key in (ord('s'), ord('S')):
            data = {
                "created_at": datetime.utcnow().isoformat()+"Z",
                "source": os.path.abspath(video_path),
                "frame_size": [editor.w, editor.h],
                "zones": [
                    {
                        "id": p["id"], 
                        "type": p.get("type", "P"),
                        "zone_name": p.get("zone_name", "parking"),
                        "points": p["points"]
                    } for p in editor.polygons
                ],
                # Keep backward compatibility with old format
                "spots": [{"id": p["id"], "points": p["points"]} for p in editor.polygons if p.get("type", "P") == "P"]
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Map saved to {out_path}")
            
            # Always save visualization image automatically with unique filename in parking_grind
            base_vis_path = out_path.replace('.json', '_zones_visualization')
            vis_path = get_unique_filename(base_vis_path, '.png')
            cv2.imwrite(vis_path, editor.show)
            print(f"[OK] Zones visualization saved to {vis_path}")
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
# Command-line interface
def _cli():
    ap = argparse.ArgumentParser(description="Parking zone editor -> JSON")
    ap.add_argument("--video", required=True, help="Path to video/image")
    ap.add_argument("--out", default="test/parking_grind/parking_map.json", help="Output JSON")
    ap.add_argument("--snap_lines", action="store_true", help="Show reference lines")
    args = ap.parse_args()
    map_parking_zones(args.video, args.out, args.snap_lines)

if __name__ == "__main__":
    _cli()
