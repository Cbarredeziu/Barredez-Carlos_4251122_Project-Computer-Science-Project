# edit_parking_zones.py
import cv2
import json
import os
import argparse
import numpy as np
from datetime import datetime

class ParkingZoneEditor:
    def __init__(self, image_path, map_path=None):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.image_path = image_path
        self.map_path = map_path
        self.display = self.image.copy()
        self.h, self.w = self.image.shape[:2]
        
        # Load existing parking zones if map exists
        self.zones = {}
        if map_path and os.path.exists(map_path):
            self.load_existing_map()
        
        # Current editing state
        self.current_zone = []
        self.selected_zone_id = None
        self.mode = "add"  # "add", "edit", "delete"
        self.zone_counter = len(self.zones) + 1
        
        # UI state
        self.show_help = True
        
    def load_existing_map(self):
        """Load existing parking map"""
        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for spot in data.get('spots', []):
                zone_id = spot['id']
                points = np.array(spot['points'], dtype=np.int32)
                self.zones[zone_id] = points
                
            print(f"[INFO] Loaded {len(self.zones)} existing parking zones")
            
            # Update counter to avoid ID conflicts
            if self.zones:
                existing_numbers = []
                for zone_id in self.zones.keys():
                    if zone_id.startswith('P') and zone_id[1:].isdigit():
                        existing_numbers.append(int(zone_id[1:]))
                if existing_numbers:
                    self.zone_counter = max(existing_numbers) + 1
                    
        except Exception as e:
            print(f"[WARN] Could not load existing map: {e}")
            self.zones = {}
    
    def get_zone_at_point(self, x, y):
        """Find which zone contains the given point"""
        point = (x, y)
        for zone_id, points in self.zones.items():
            if cv2.pointPolygonTest(points, point, False) >= 0:
                return zone_id
        return None
    
    def draw_zones(self):
        """Draw all parking zones on the display image"""
        self.display = self.image.copy()
        
        # Draw existing zones
        for zone_id, points in self.zones.items():
            color = (0, 255, 0) if zone_id != self.selected_zone_id else (0, 255, 255)
            thickness = 2 if zone_id != self.selected_zone_id else 3
            
            # Draw polygon
            cv2.polylines(self.display, [points], True, color, thickness)
            
            # Fill with semi-transparent color
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(self.display, 0.8, overlay, 0.2, 0, self.display)
            
            # Add zone label
            center = np.mean(points, axis=0).astype(int)
            cv2.putText(self.display, zone_id, (center[0]-10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current zone being created
        if len(self.current_zone) > 0:
            if len(self.current_zone) > 1:
                cv2.polylines(self.display, [np.array(self.current_zone)], False, (255, 0, 0), 2)
            for i, point in enumerate(self.current_zone):
                cv2.circle(self.display, point, 5, (255, 0, 0), -1)
                cv2.putText(self.display, str(i+1), (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw help text
        if self.show_help:
            self.draw_help_text()
    
    def draw_help_text(self):
        """Draw help text on the display"""
        help_text = [
            f"Mode: {self.mode.upper()}",
            "",
            "CONTROLS:",
            "Left Click: Add point / Select zone",
            "Right Click: Finish zone / Deselect",
            "SPACE: Change mode (Add/Edit/Delete)",
            "D: Delete selected zone",
            "S: Save parking map",
            "R: Reset current zone",
            "H: Toggle help",
            "ESC/Q: Exit",
            "",
            f"Zones: {len(self.zones)} | Current: {self.selected_zone_id or 'None'}"
        ]
        
        # Draw semi-transparent background
        cv2.rectangle(self.display, (10, 10), (300, len(help_text) * 20 + 20), (0, 0, 0), -1)
        cv2.rectangle(self.display, (10, 10), (300, len(help_text) * 20 + 20), (255, 255, 255), 1)
        
        for i, text in enumerate(help_text):
            color = (0, 255, 255) if text.startswith("Mode:") else (255, 255, 255)
            cv2.putText(self.display, text, (15, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "add":
                self.current_zone.append((x, y))
                print(f"Added point {len(self.current_zone)}: ({x}, {y})")
                
            elif self.mode == "edit" or self.mode == "delete":
                zone_id = self.get_zone_at_point(x, y)
                if zone_id:
                    self.selected_zone_id = zone_id
                    print(f"Selected zone: {zone_id}")
                else:
                    self.selected_zone_id = None
                    print("No zone selected")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "add" and len(self.current_zone) >= 3:
                # Finish current zone
                zone_id = f"P{self.zone_counter}"
                self.zones[zone_id] = np.array(self.current_zone, dtype=np.int32)
                print(f"Created zone {zone_id} with {len(self.current_zone)} points")
                self.current_zone = []
                self.zone_counter += 1
            else:
                # Deselect or cancel
                self.selected_zone_id = None
                self.current_zone = []
    
    def save_map(self, output_path):
        """Save the parking map to JSON file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create the parking map data structure
            parking_data = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source": os.path.abspath(self.image_path),
                "frame_size": [self.w, self.h],
                "spots": []
            }
            
            for zone_id, points in self.zones.items():
                spot_data = {
                    "id": zone_id,
                    "points": points.tolist()
                }
                parking_data["spots"].append(spot_data)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parking_data, f, ensure_ascii=False, indent=2)
            
            print(f"[OK] Saved {len(self.zones)} zones to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Could not save map: {e}")
            return False
    
    def delete_selected_zone(self):
        """Delete the currently selected zone"""
        if self.selected_zone_id and self.selected_zone_id in self.zones:
            del self.zones[self.selected_zone_id]
            print(f"Deleted zone: {self.selected_zone_id}")
            self.selected_zone_id = None
        else:
            print("No zone selected for deletion")
    
    def cycle_mode(self):
        """Cycle through editing modes"""
        modes = ["add", "edit", "delete"]
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        print(f"Switched to {self.mode.upper()} mode")
        
        # Reset state when changing modes
        self.current_zone = []
        self.selected_zone_id = None
    
    def run(self):
        """Main editing loop"""
        cv2.namedWindow("Parking Zone Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Parking Zone Editor", self.mouse_callback)
        
        print("\n=== Parking Zone Editor ===")
        print("Left click to add points or select zones")
        print("Right click to finish zone or deselect")
        print("Press SPACE to change mode, S to save, Q to quit")
        print(f"Loaded {len(self.zones)} existing zones")
        
        while True:
            self.draw_zones()
            cv2.imshow("Parking Zone Editor", self.display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key in [27, ord('q'), ord('Q')]:  # ESC or Q
                break
            elif key == ord(' '):  # SPACE - change mode
                self.cycle_mode()
            elif key == ord('s') or key == ord('S'):  # Save
                if self.map_path:
                    self.save_map(self.map_path)
                else:
                    # Default save location
                    default_path = "test/parking_map.json"
                    self.save_map(default_path)
            elif key == ord('d') or key == ord('D'):  # Delete
                self.delete_selected_zone()
            elif key == ord('r') or key == ord('R'):  # Reset current zone
                self.current_zone = []
                print("Reset current zone")
            elif key == ord('h') or key == ord('H'):  # Toggle help
                self.show_help = not self.show_help
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Interactive Parking Zone Editor")
    parser.add_argument("--image", required=True, help="Path to parking lot image")
    parser.add_argument("--map", help="Path to existing parking map JSON (optional)")
    parser.add_argument("--output", default="test/parking_map.json", help="Output path for parking map")
    
    args = parser.parse_args()
    
    # Use existing map if not specified but exists in default location
    map_path = args.map
    if not map_path and os.path.exists(args.output):
        map_path = args.output
        print(f"[INFO] Using existing map: {map_path}")
    
    try:
        editor = ParkingZoneEditor(args.image, map_path)
        editor.map_path = args.output  # Set output path
        editor.run()
        
        # Final save prompt
        if editor.zones:
            print(f"\nFinal save to {args.output}?")
            response = input("Save changes? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                editor.save_map(args.output)
            else:
                print("Changes not saved")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()