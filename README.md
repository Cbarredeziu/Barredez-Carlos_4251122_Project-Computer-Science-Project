# Real-Time Parking Occupancy Detection System

## Overview
This project implements a computer vision-based parking occupancy detection system that can monitor parking spaces in real-time using video feeds or static images. The system uses YOLO (You Only Look Once) object detection to identify vehicles and determine parking space occupancy status without requiring intrusive hardware installations.

## Problem Statement
In dense urban areas, on-street parking demand exceeds supply, and current meter systems often fail to align paid time with actual use. This leads to low turnover, congestion from cruising for parking, revenue leakage, and high operating costs due to manual enforcement. A computer-vision–based management system is needed to measure real-time occupancy per bay without adding physical obstacles, while collecting anonymized usage data to generate occupancy analytics and enable data-driven urban planning.

## Key Features
- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection (cars, motorcycles, buses, trucks)
- **Interactive Zone Mapping**: Graphical tool for defining parking space boundaries on video/image frames
- **Occupancy Analysis**: Determines parking space status based on vehicle-zone overlap
- **Temporal Smoothing**: Reduces false positives through frame-based voting system
- **Multiple Input Support**: Works with both video files and static images
- **Data Export**: Generates CSV logs and JSON summaries for further analysis
- **Visual Feedback**: Real-time visualization with colored overlays and status indicators

## System Architecture

### Core Components
1. **Zone Mapping Tool** (`map_parking_zones.py`)
   - Interactive polygon editor for defining parking spaces
   - Line detection assistance for reference alignment
   - JSON export of parking zone coordinates

2. **Occupancy Detection Engine** (`detect_occupancy.py`)
   - YOLO-based vehicle detection
   - Overlap calculation between detected vehicles and parking zones
   - Temporal smoothing for stable occupancy status
   - Real-time visualization and data logging

3. **Main Pipeline** (`main.py`)
   - Orchestrates the complete workflow
   - Automatically handles zone mapping if not present
   - Configurable detection parameters

### Technical Stack
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Machine Learning**: PyTorch (YOLO backend)
- **Data Processing**: NumPy, Pandas
- **Output Formats**: JSON, CSV, PNG visualizations
- **Language**: Python 3.8+

## Installation and Setup

### Prerequisites
```bash
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install ultralytics>=8.0.0
pip install torch>=2.0.0
pip install pillow>=10.0.0
pip install matplotlib>=3.7.0
```

### Quick Start
1. **Clone the repository**
2. **Install dependencies**: `pip install -r src/requirements.txt`
3. **Run the system**: `python src/main.py --video [path_to_video_or_image] --display`

## Usage

### Basic Operation
```bash
# Process a video with interactive display
python src/main.py --video parking_video.mp4 --display

# Process an image and save visualization
python src/main.py --video parking_image.jpg --save_image output.png

# Use custom YOLO weights and detection parameters
python src/main.py --video input.mp4 --weights yolov8x.pt --conf 0.5 --overlap_thr 0.8
```

### Zone Mapping (First Time Setup)
1. Run the main script with a video/image file
2. If no parking map exists, the zone editor will open automatically
3. Click to define parking space boundaries (polygons)
4. Press ENTER to close each polygon
5. Press 'S' to save the mapping JSON
6. Press 'Q' to exit and proceed to detection

### Detection Parameters
- `--conf`: Detection confidence threshold (default: 0.35)
- `--overlap_thr`: Minimum overlap to consider a space occupied (default: 0.15)
- `--history`: Temporal smoothing window size (default: 5 frames)
- `--weights`: YOLO model weights (default: yolov8n.pt)

## Output Data

### JSON Summary (`occupancy_lastframe.json`)
```json
{
  "ts_utc": "2025-11-10T13:32:19.362570Z",
  "frame": 1,
  "spots": {
    "P1": {"status": "free", "overlap": 0.0},
    "P2": {"status": "occupied", "overlap": 0.984}
  }
}
```

### CSV Log (`occupancy_log.csv`)
Contains timestamped records with:
- Timestamp (UTC)
- Frame number
- Spot ID
- Occupancy status (free/occupied)
- Vehicle type and confidence
- Overlap percentage

### Visualizations
- Real-time overlay showing parking zones (green=free, red=occupied)
- Vehicle bounding boxes with detection confidence
- Occupancy counters and status information

## Current Phase Status
**Phase 2 - Prototype Implementation Complete**
- ✅ Core detection system implemented
- ✅ Interactive zone mapping tool
- ✅ Real-time visualization
- ✅ Data logging and export
- ✅ Support for both video and image inputs
- ✅ Temporal smoothing for stability
- ✅ Vehicle tracking with unique IDs

**Next Phase Goals:**
- Integration with real-time camera streams (RTSP)
- Database integration for persistent storage
- Web-based dashboard for monitoring
- API development for external integrations
- Performance optimization for edge deployment

## Technical Considerations

### Detection Accuracy
The system achieves reliable occupancy detection through:
- Pre-trained YOLO models optimized for vehicle detection
- Configurable overlap thresholds to handle partial occlusions
- Temporal voting to reduce noise and false positives
- Support for multiple vehicle classes (cars, motorcycles, buses, trucks)

### Privacy and Compliance
- No storage of vehicle identification data (license plates, faces)
- Anonymous vehicle tracking with temporary IDs
- Local processing without cloud dependencies
- Configurable data retention policies

### Performance Optimization
- Efficient polygon-based zone checking
- Optimized OpenCV operations for real-time processing
- Scalable architecture suitable for edge deployment
- Memory-efficient frame processing
