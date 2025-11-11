# edit_parking_zones.py - Enhanced Multi-Zone Parking Editor with ID Management
import cv2
import json
import os
import argparse
import numpy as np
from datetime import datetime

class EnhancedParkingZoneEditor:
    def __init__(self, image_path, map_path=None):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.image_path = image_path
        self.map_path = map_path
        self.display = self.image.copy()
        self.h, self.w = self.image.shape[:2]
        
        # Enhanced zone types with colors and prefixes
        self.zone_types = {
            "P": {"name": "parking", "color": (0, 255, 0), "prefix": "P"},      # Green for parking
            "T": {"name": "traffic", "color": (0, 165, 255), "prefix": "T"},    # Orange for traffic  
            "N": {"name": "no_parking", "color": (0, 0, 255), "prefix": "N"},   # Red for no-parking
            "D": {"name": "disabled", "color": (255, 0, 0), "prefix": "D"}      # Blue for disabled (accessibility standard)
        }
        
        # Initialize editing state first
        self.zones = {}
        self.current_zone = []
        self.selected_zone_id = None
        self.mode = "add"  # "add", "edit", "delete", "rename"
        self.current_zone_type = "P"  # Default to parking
        self.zone_counters = {"P": 1, "T": 1, "N": 1, "D": 1}
        
        # UI state
        self.show_help = True
        self.input_mode = None  # None, "rename"
        self.input_text = ""
        self.has_saved = False  # Track if user has saved during session
        
        # Load existing parking zones if map exists
        if map_path and os.path.exists(map_path):
            self.load_existing_map()
        
        # Update counters after loading existing zones
        self.update_zone_counters()
        
        # UI state
        self.show_help = True
        self.input_text = ""
        self.input_mode = None  # None, "rename"
        
    def load_existing_map(self):
        """Load existing parking map with multi-zone support"""
        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Support both old and new formats
            if "zones" in data:
                # New multi-zone format
                for zone in data["zones"]:
                    zone_id = zone['id']
                    points = np.array(zone['points'], dtype=np.int32)
                    zone_type = zone.get('type', 'P')
                    self.zones[zone_id] = {
                        "points": points,
                        "type": zone_type,
                        "zone_name": zone.get("zone_name", self.zone_types[zone_type]["name"])
                    }
                print(f"[INFO] Loaded {len(self.zones)} multi-zone entries")
            else:
                # Old format - convert to new format
                for spot in data.get('spots', []):
                    zone_id = spot['id']
                    points = np.array(spot['points'], dtype=np.int32)
                    self.zones[zone_id] = {
                        "points": points,
                        "type": "P",
                        "zone_name": "parking"
                    }
                print(f"[INFO] Loaded {len(self.zones)} zones (converted from old format)")
            
            # Update zone counters to avoid ID conflicts
            self.update_zone_counters()
                    
        except Exception as e:
            print(f"[WARN] Could not load existing map: {e}")
            self.zones = {}
    
    def update_zone_counters(self):
        """Update zone counters based on existing zones"""
        for zone_type in self.zone_counters:
            existing_numbers = []
            for zone_id in self.zones.keys():
                if zone_id.startswith(zone_type) and len(zone_id) > 1 and zone_id[1:].isdigit():
                    existing_numbers.append(int(zone_id[1:]))
            if existing_numbers:
                self.zone_counters[zone_type] = max(existing_numbers) + 1
    
    def get_zone_at_point(self, x, y):
        """Find which zone contains the given point"""
        point = (x, y)
        for zone_id, zone_data in self.zones.items():
            points = zone_data["points"]
            if cv2.pointPolygonTest(points, point, False) >= 0:
                return zone_id
        return None
    
    def change_zone_id(self, old_id, new_id):
        """Change the ID of a zone"""
        if old_id not in self.zones:
            print(f"Zone {old_id} not found")
            return False
        
        if new_id in self.zones:
            print(f"Zone ID {new_id} already exists")
            return False
        
        if not new_id.strip():
            print("Zone ID cannot be empty")
            return False
        
        # Move zone data to new ID
        self.zones[new_id] = self.zones[old_id]
        del self.zones[old_id]
        
        # Update selected zone if it was the renamed one
        if self.selected_zone_id == old_id:
            self.selected_zone_id = new_id
        
        print(f"Renamed zone {old_id} to {new_id}")
        return True
    
    def draw_zones(self):
        """Draw all parking zones with color-coded types"""
        self.display = self.image.copy()
        
        # Draw existing zones with type-specific colors
        for zone_id, zone_data in self.zones.items():
            points = zone_data["points"]
            zone_type = zone_data["type"]
            
            # Get zone color and adjust for selection
            base_color = self.zone_types[zone_type]["color"]
            if zone_id == self.selected_zone_id:
                color = (255, 255, 0)  # Yellow for selected
                thickness = 3
            else:
                color = base_color
                thickness = 2
            
            # Draw polygon
            cv2.polylines(self.display, [points], True, color, thickness)
            
            # Fill with semi-transparent color
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(self.display, 0.8, overlay, 0.2, 0, self.display)
            
            # Add zone label with type indicator
            center = np.mean(points, axis=0).astype(int)
            label = f"{zone_id}({zone_type})"
            cv2.putText(self.display, label, (center[0]-20, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current zone being created with current type color
        if len(self.current_zone) > 0:
            current_color = self.zone_types[self.current_zone_type]["color"]
            if len(self.current_zone) > 1:
                cv2.polylines(self.display, [np.array(self.current_zone)], False, current_color, 2)
            for i, point in enumerate(self.current_zone):
                cv2.circle(self.display, point, 5, current_color, -1)
                cv2.putText(self.display, str(i+1), (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)
        
        # Draw zone type indicator
        self.draw_zone_type_indicator()
        
        # Draw help text
        if self.show_help:
            self.draw_help_text()
        
        # Draw input text if in input mode
        if self.input_mode:
            self.draw_input_text()
    
    def draw_zone_type_indicator(self):
        """Draw zone type selection indicator"""
        y_start = self.h - 100
        for i, (zone_type, config) in enumerate(self.zone_types.items()):
            x_pos = 20 + i * 80
            color = config["color"]
            
            # Highlight current type
            if zone_type == self.current_zone_type:
                cv2.rectangle(self.display, (x_pos-5, y_start-5), (x_pos+65, y_start+45), (255, 255, 255), 2)
            
            # Draw type indicator
            cv2.rectangle(self.display, (x_pos, y_start), (x_pos+60, y_start+40), color, -1)
            cv2.rectangle(self.display, (x_pos, y_start), (x_pos+60, y_start+40), (255, 255, 255), 1)
            
            # Add text
            cv2.putText(self.display, f"{i+1}:{zone_type}", (x_pos+5, y_start+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(self.display, config["name"][:6], (x_pos+5, y_start+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    def draw_help_text(self):
        """Draw enhanced help text"""
        help_text = [
            f"Mode: {self.mode.upper()} | Zone Type: {self.current_zone_type} ({self.zone_types[self.current_zone_type]['name']})",
            "",
            "ZONE TYPES (Number Keys):",
            "1: Parking (Green) | 2: Traffic (Orange)",
            "3: No-parking (Red) | 4: Disabled (Blue)",
            "",
            "CONTROLS:",
            "Left Click: Add point / Select zone",
            "Right Click: Finish zone / Deselect",
            "SPACE: Change mode (Add/Edit/Delete/Rename)",
            "I: Change ID of selected zone",
            "D: Delete selected zone",
            "S: Save parking map",
            "R: Reset current zone",
            "H: Toggle help",
            "ESC/Q: Exit",
            "",
            f"Zones: {len(self.zones)} | Selected: {self.selected_zone_id or 'None'}"
        ]
        
        # Draw semi-transparent background
        cv2.rectangle(self.display, (10, 10), (400, len(help_text) * 18 + 20), (0, 0, 0), -1)
        cv2.rectangle(self.display, (10, 10), (400, len(help_text) * 18 + 20), (255, 255, 255), 1)
        
        for i, text in enumerate(help_text):
            if text.startswith("Mode:"):
                color = (0, 255, 255)
            elif "TYPES" in text or "CONTROLS" in text:
                color = (255, 255, 0)
            else:
                color = (255, 255, 255)
            cv2.putText(self.display, text, (15, 25 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def draw_input_text(self):
        """Draw input text box for ID changes"""
        if self.input_mode == "rename":
            # Draw input box
            box_w, box_h = 300, 60
            box_x = (self.w - box_w) // 2
            box_y = (self.h - box_h) // 2
            
            cv2.rectangle(self.display, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            cv2.rectangle(self.display, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
            
            # Draw text
            cv2.putText(self.display, "Enter new ID:", (box_x + 10, box_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(self.display, self.input_text + "_", (box_x + 10, box_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle enhanced mouse events with multi-zone support"""
        if self.input_mode:  # Skip mouse events when in input mode
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "add":
                self.current_zone.append((x, y))
                print(f"Added point {len(self.current_zone)}: ({x}, {y}) for {self.current_zone_type} zone")
                
            elif self.mode in ["edit", "delete", "rename"]:
                zone_id = self.get_zone_at_point(x, y)
                if zone_id:
                    self.selected_zone_id = zone_id
                    zone_type = self.zones[zone_id]["type"]
                    print(f"Selected zone: {zone_id} (Type: {zone_type})")
                else:
                    self.selected_zone_id = None
                    print("No zone selected")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "add" and len(self.current_zone) >= 3:
                # Finish current zone with type
                zone_id = f"{self.current_zone_type}{self.zone_counters[self.current_zone_type]}"
                self.zones[zone_id] = {
                    "points": np.array(self.current_zone, dtype=np.int32),
                    "type": self.current_zone_type,
                    "zone_name": self.zone_types[self.current_zone_type]["name"]
                }
                print(f"Created {self.current_zone_type} zone {zone_id} with {len(self.current_zone)} points")
                self.current_zone = []
                self.zone_counters[self.current_zone_type] += 1
            else:
                # Deselect or cancel
                self.selected_zone_id = None
                self.current_zone = []
    
    def save_map(self, output_path):
        """Save the enhanced parking map with multi-zone support"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create enhanced parking map data structure
            parking_data = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source": os.path.abspath(self.image_path),
                "frame_size": [self.w, self.h],
                "zones": [],
                # Keep backward compatibility
                "spots": []
            }
            
            for zone_id, zone_data in self.zones.items():
                # New enhanced format
                zone_entry = {
                    "id": zone_id,
                    "points": zone_data["points"].tolist(),
                    "type": zone_data["type"],
                    "zone_name": zone_data["zone_name"]
                }
                parking_data["zones"].append(zone_entry)
                
                # Backward compatibility for parking zones
                if zone_data["type"] in ["P", "D"]:
                    spot_data = {
                        "id": zone_id,
                        "points": zone_data["points"].tolist()
                    }
                    parking_data["spots"].append(spot_data)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parking_data, f, ensure_ascii=False, indent=2)
            
            # Also save visualization
            vis_path = output_path.replace('.json', '_zones_visualization.png')
            self.save_visualization(vis_path)
            
            print(f"[OK] Saved {len(self.zones)} zones to: {output_path}")
            print(f"[OK] Saved visualization to: {vis_path}")
            self.has_saved = True  # Mark that we've saved during this session
            return True
            
        except Exception as e:
            print(f"[ERROR] Could not save map: {e}")
            return False
    
    def save_visualization(self, vis_path):
        """Save current visualization as PNG"""
        try:
            # Create clean visualization without help text
            temp_help = self.show_help
            temp_input = self.input_mode
            self.show_help = False
            self.input_mode = None
            
            self.draw_zones()
            cv2.imwrite(vis_path, self.display)
            
            # Restore UI state
            self.show_help = temp_help
            self.input_mode = temp_input
            
        except Exception as e:
            print(f"[WARN] Could not save visualization: {e}")
    
    def delete_selected_zone(self):
        """Delete the currently selected zone"""
        if self.selected_zone_id and self.selected_zone_id in self.zones:
            zone_type = self.zones[self.selected_zone_id]["type"]
            del self.zones[self.selected_zone_id]
            print(f"Deleted {zone_type} zone: {self.selected_zone_id}")
            self.selected_zone_id = None
        else:
            print("No zone selected for deletion")
    
    def cycle_mode(self):
        """Cycle through enhanced editing modes"""
        modes = ["add", "edit", "delete", "rename"]
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        print(f"Switched to {self.mode.upper()} mode")
        
        # Reset state when changing modes
        self.current_zone = []
        self.selected_zone_id = None
        self.input_mode = None
        self.input_text = ""
    
    def handle_zone_type_key(self, key):
        """Handle zone type selection keys (1-4)"""
        type_map = {ord('1'): 'P', ord('2'): 'T', ord('3'): 'N', ord('4'): 'D'}
        if key in type_map:
            self.current_zone_type = type_map[key]
            zone_name = self.zone_types[self.current_zone_type]["name"]
            print(f"Selected zone type: {self.current_zone_type} ({zone_name})")
            return True
        return False
    
    def handle_text_input(self, key):
        """Handle text input for ID changes"""
        if self.input_mode != "rename":
            return False
            
        if key == 13:  # Enter
            if self.input_text.strip() and self.selected_zone_id:
                success = self.change_zone_id(self.selected_zone_id, self.input_text.strip())
                if success:
                    self.input_mode = None
                    self.input_text = ""
            else:
                print("Invalid input or no zone selected")
        elif key == 27:  # Escape
            self.input_mode = None
            self.input_text = ""
            print("ID change cancelled")
        elif key == 8:  # Backspace
            self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126:  # Printable characters
            self.input_text += chr(key)
        
        return True
    
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
    parser = argparse.ArgumentParser(description="Enhanced Multi-Zone Parking Editor")
    parser.add_argument("--image", required=True, help="Path to parking lot image")
    parser.add_argument("--map", help="Path to existing parking map JSON (optional)")
    parser.add_argument("--output", default="test/parking_map.json", help="Output path for parking map")
    
    args = parser.parse_args()
    
    print("Enhanced Parking Zone Editor - Multi-Zone Support")
    print("=================================================")
    print("Zone Types:")
    print("  1 = Parking (Green)")
    print("  2 = Traffic (Orange)")  
    print("  3 = No-parking (Red)")
    print("  4 = Disabled (Blue)")
    print()
    print("Controls:")
    print("  Keys 1-4: Select zone type")
    print("  Left Click: Add point to zone")
    print("  Right Click: Complete zone")
    print("  SPACE: Cycle modes (ADD/EDIT/DELETE/RENAME)")
    print("  S: Save map")
    print("  D: Delete selected zone")
    print("  R: Reset current zone")
    print("  H: Toggle help display")
    print("  Q/ESC: Quit")
    print("=================================================")
    print()
    
    # Use existing map if not specified but exists in default location
    map_path = args.map
    if not map_path and os.path.exists(args.output):
        map_path = args.output
        print(f"[INFO] Using existing map: {map_path}")
    
    try:
        editor = EnhancedParkingZoneEditor(args.image, map_path)
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