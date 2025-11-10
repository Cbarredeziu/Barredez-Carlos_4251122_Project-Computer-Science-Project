# map_parking_zones.py
import cv2
import json
import os
import argparse
import numpy as np
from datetime import datetime

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
        self.polygons = []  # list of {"id": str, "points": [(x,y),...]}
        self.curr_pts = []
        self.next_id = 1

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.curr_pts.append((x,y))
            self.redraw()

    def redraw(self):
        self.show = self.base.copy()
        # saved polygons
        for poly in self.polygons:
            pts = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(self.show, [pts], isClosed=True, thickness=2, color=(0,255,0))
            c = pts.mean(axis=0).astype(int)
            cv2.putText(self.show, poly["id"], c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        # polygon being edited
        if self.curr_pts:
            pts = np.array(self.curr_pts, dtype=np.int32)
            cv2.polylines(self.show, [pts], isClosed=False, thickness=2, color=(0,200,255))
            for p in self.curr_pts:
                cv2.circle(self.show, p, 4, (0,200,255), -1)

        # HUD
        cv2.rectangle(self.show, (0,0), (self.w, 30), (0,0,0), -1)
        hud = "[LMB] points  [ENTER] close  [U] undo  [N] new  [S] save JSON  [Q] quit"
        cv2.putText(self.show, hud, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    def close_current(self):
        if len(self.curr_pts) >= 3:
            pid = f"P{self.next_id}"
            self.polygons.append({"id": pid, "points": self.curr_pts.copy()})
            self.next_id += 1
        self.curr_pts = []
        self.redraw()

def map_parking_zones(video_path, out_path="parking_map.json", snap_lines=False):
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
        elif key in (ord('s'), ord('S')):
            data = {
                "created_at": datetime.utcnow().isoformat()+"Z",
                "source": os.path.abspath(video_path),
                "frame_size": [editor.w, editor.h],
                "spots": [{"id": p["id"], "points": p["points"]} for p in editor.polygons]
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Map saved to {out_path}")
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()

def _cli():
    ap = argparse.ArgumentParser(description="Parking zone editor -> JSON")
    ap.add_argument("--video", required=True, help="Path to video/image")
    ap.add_argument("--out", default="parking_map.json", help="Output JSON")
    ap.add_argument("--snap_lines", action="store_true", help="Show reference lines")
    args = ap.parse_args()
    map_parking_zones(args.video, args.out, args.snap_lines)

if __name__ == "__main__":
    _cli()
