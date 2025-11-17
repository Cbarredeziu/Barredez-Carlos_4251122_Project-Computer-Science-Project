# Parking control system using computer vision and convolutional neural network models for object detection.

## Overview
This project implements an intelligent computer vision-based parking occupancy detection system that monitors parking spaces in real-time using video feeds or static images. The system uses YOLOv8 for accurate vehicle detection and features multi-zone support for complex parking environments including traffic lanes, no-parking zones, and accessibility compliance.

## Problem Statement
In dense urban areas, on-street parking demand exceeds supply, and current meter systems often fail to align paid time with actual use. This leads to low turnover, congestion from cruising for parking, revenue leakage, and high operating costs due to manual enforcement. A computer-vision–based management system is needed to measure real-time occupancy per bay without adding physical obstacles, while collecting anonymized usage data to generate occupancy analytics and enable data-driven urban planning.

## Key Features
- **Advanced Vehicle Detection**: Uses YOLOv11 for accurate detection (cars, motorcycles, buses, trucks)
- **Multi-Zone Support**: Supports 4 zone types (Parking, Traffic, No-parking, Disabled)
- **Interactive Zone Mapping**: Graphical tool with zone type selection and color coding
- **Smart Vehicle Categorization**: 5 status categories (Parked, Passing, Illegal, Partial, Unassigned)
- **Accessibility Compliance**: Proper blue coloring for disabled parking zones
- **Grid Analysis Mode**: Simplified zone counting for layout analysis
- **Temporal Smoothing**: Reduces false positives through frame-based voting system
- **Folder-Based Processing**: Automatically processes all images in organized input folders
- **Consolidated Data Export**: Single CSV and JSON files containing all results with source identification
- **Visual Feedback**: Real-time visualization with colored overlays and status indicators
- **Modular Architecture**: Split into maintainable utility and detection modules

## System Architecture

### Core Components
1. **Zone Mapping Tool** (`map_parking_zones.py`)
   - Interactive polygon editor with multi-zone support
   - Zone type selection (Keys 1-4: P/T/N/D)
   - Color-coded visualization (Green/Orange/Red/Blue)
   - Line detection assistance for reference alignment
   - JSON export with zone type metadata

2. **Parking Utilities** (`parking_utils.py`)
   - Zone loading and mapping functions with backward compatibility
   - Geometric calculations (masks, areas, overlaps)
   - Data smoothing and status voting algorithms
   - File utilities and helper functions
   - Constants and vehicle class definitions

3. **Vehicle Detection Engine** (`vehicle_detector.py`)
   - YOLO-based vehicle detection with tracking
   - Multi-zone overlap calculation and status assignment
   - Advanced visualization with status-based colors
   - Consolidated CSV/JSON logging with source identification
   - Grid analysis vs regular detection modes

4. **Evaluation & Analytics** (`analyze_occupancy_data.py`)
   - **Ground truth comparison** against test configurations
   - **Binary classification metrics**: Accuracy, Precision, Recall, F1-Score
   - **Multi-class vehicle type metrics**: Per-class and macro/weighted averages
   - **Confusion matrices**: Binary (Occupied vs Free) and Multi-class (Vehicle Types)
   - **Performance visualization**: Metrics charts, error analysis, per-class breakdown
   - **Comprehensive reporting**: JSON metrics export and detailed console summaries

5. **Main Pipeline** (`main.py`)
   - **Dual-mode operation**: Normal detection + Interactive editor
   - Smart directory detection (works from root or src)
   - Automatically handles zone mapping if not present
   - Folder-based batch processing with dynamic paths
   - Configurable detection parameters
   - Enhanced parking zone editor launcher

### Technical Stack
- **Computer Vision**: OpenCV, YOLOv11 (Ultralytics)
- **Machine Learning**: PyTorch (YOLO backend)
- **Data Processing**: NumPy, Pandas
- **Evaluation Metrics**: scikit-learn (Classification metrics, Confusion matrices)
- **Visualization**: Matplotlib, Seaborn (Enhanced heatmaps)
- **Output Formats**: JSON, CSV, PNG visualizations
- **Architecture**: Modular design with separated concerns
- **Language**: Python 3.8+

### Zone Types & Color Coding
| Zone Type | Code | Purpose | Color | Usage |
|-----------|------|---------|-------|-------|
| **Parking** | P | Regular parking spaces | Green | Standard vehicle parking |
| **Traffic** | T | Traffic/driving lanes | Orange | Vehicles passing through |
| **No-Parking** | N | Prohibited parking areas | Red | Violation detection |
| **Disabled** | D | Accessibility spaces | Blue | ADA compliant parking |

### Vehicle Status Categories
| Status | Description | Visual Indicator |
|--------|-------------|-----------------|
| **Parked** | Properly parked in designated zone | White box |
| **Passing** | Vehicle in traffic zone (temporary) | Orange box [TRAFFIC] |
| **Illegal** | Vehicle in no-parking zone | Red box [NO-PARK] |
| **Partial** | Vehicle overlaps parking zone boundary | Yellow box [PARTIAL] |
| **Unassigned** | Vehicle not in any defined zone | Red box [ILLEGAL] |

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
pip install scikit-learn>=1.3.0
pip install seaborn>=0.12.0
```

### Quick Start
1. **Clone the repository**
2. **Install dependencies**: `pip install -r src/requirements.txt`
3. **Add your images**: 
   - Place reference images (empty parking lots) in `data/grid_photos/`
   - Place detection images (with cars) in `data/inputs/`
4. **Run the system**: 
   - From root: `python src/main.py`
   - From src: `python main.py`

## Folder Structure

### Input Data Folders
```
data/
├── inputs/               # Images/videos with cars for occupancy detection
└── grid_photos/          # Reference images for zone mapping (empty parking lots)
```

### Configuration & Results
```
test/
├── parking_grind/        # Parking zone configuration files
│   ├── parking_map.json
│   └── parking_map_zones_visualization.png
└── results/              # All output files (see below)
```

### Output Folders
```
test/results/
├── images/               # Visualization images for all processed files
├── data/                 # Consolidated CSV and JSON data files
│   ├── occupancy_log.csv           # All input detection results
│   ├── occupancy_summary.json      # All input summaries by file
│   ├── grid_analysis_log.csv       # All grid analysis results
│   ├── grid_analysis_summary.json  # All grid summaries by file
│   ├── matched_detections_*.csv    # Ground truth comparison results
│   ├── evaluation_metrics_*.json   # Evaluation metrics (F1, precision, recall)
│   └── plots/                      # Evaluation visualizations
│       ├── confusion_matrix_binary_*.png      # Binary classification matrix
│       ├── confusion_matrix_multiclass_*.png  # Vehicle type classification matrix
│       ├── performance_metrics_*.png          # Metrics comparison charts
│       ├── per_class_metrics_*.png            # Per-vehicle-type performance
│       └── error_analysis_*.png               # TP/TN/FP/FN breakdown
└── grid_results/         # Grid-specific visualization images
```

## Usage

### Dual-Mode Operation

**Detection Mode (Default):**
```bash
# Simple run - processes all files automatically
python src/main.py                    # From root directory
python main.py                        # From src directory

# With display windows and custom parameters
python src/main.py --display --conf 0.5 --overlap_thr 0.8

# Disable image saving (saves processing time)
python src/main.py --no_save_image
```

**Interactive Editor Mode:**
```bash
# Launch parking zone editor
python src/main.py --edit --image data/inputs/parkingplace_cars1.png

# Edit existing zones
python src/main.py --edit --image data/grid_photos/parkingplace_grid.png
```

### Enhanced Zone Mapping

**Automatic Mode (First Time Setup):**
1. Place a reference image (empty parking lot) in `data/grid_photos/`
2. Run `python src/main.py`
3. The zone editor will open automatically if no parking map exists

**Manual Mode (Anytime):**
```bash
python src/main.py --edit --image data/grid_photos/reference_image.png
```

**Editor Controls:**
- **Keys 1-4**: Select zone type (P=Parking, T=Traffic, N=No-parking, D=Disabled)
- **Left Click**: Add points to define zone boundaries
- **Right Click**: Complete current zone polygon
- **SPACE**: Cycle editing modes (ADD/EDIT/DELETE/RENAME)
- **S**: Save zones and visualization
- **H**: Toggle help display
- **Q/ESC**: Exit editor

**Advanced Features:**
- **Smart save tracking**: No redundant save prompts
- **Zone ID management**: Change zone IDs in RENAME mode
- **Visual feedback**: Real-time zone highlighting and type indicators
- **Color-coded editing**: Each zone type has distinct colors

### Detection Modes

**Grid Analysis Mode** (`data/grid_photos/`):
- Shows zone counts only (Parking: X, Traffic: Y, etc.)
- No individual vehicle status labels
- Single output file (overwrites previous)
- Perfect for layout planning and zone verification

**Regular Detection Mode** (`data/inputs/`):
- Full vehicle status information with labels
- Detailed vehicle categorization (Parked/Passing/Illegal/etc.)
- Unique output filenames with timestamps
- Complete occupancy monitoring and violation detection

**Path Flexibility:**
- Works from both root directory (`python src/main.py`)
- Works from src directory (`python main.py`)
- Smart directory detection and path resolution
- Cross-platform compatibility (Windows/Linux/Mac)

### Regular Workflow

1. **Setup zones once**: 
   - Place reference image in `data/grid_photos/`
   - Run `python src/main.py` (auto-launches editor if no map exists)
   - Or manually: `python src/main.py --edit --image data/grid_photos/reference.png`

2. **Add detection images**: Place images with cars in `data/inputs/`

3. **Run analysis**: Execute `python src/main.py` from any directory

4. **Review results**: Check `test/results/` for all outputs:
   - **Images**: `test/results/images/` and `test/results/grid_results/`
   - **Data**: `test/results/data/` (CSV and JSON files)

5. **Edit zones anytime**: Use `--edit` mode to modify existing zones

### Evaluation and Testing

**Run System Evaluation:**
```bash
# Evaluate detection performance against ground truth test configuration
python src/analyze_occupancy_data.py --csv test/results/data/occupancy_log.csv --test_config test/test_config_20_tests.json --output_dir test/results/data
```

**Evaluation Features:**
- **Binary Classification Metrics**: Measures occupancy detection accuracy (Free vs Occupied)
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix with TP/TN/FP/FN breakdown
  
- **Multi-class Classification Metrics**: Evaluates vehicle type detection
  - Per-class metrics (Car, Truck, Bus, Motorcycle)
  - Macro and weighted F1-scores
  - Detailed confusion matrix for all vehicle types

- **Visual Analytics**:
  - Confusion matrix heatmaps (binary and multi-class)
  - Performance metrics comparison charts
  - Per-class precision/recall/F1-score breakdown
  - Error analysis visualization

**Test Configuration Format:**
Create test cases in JSON format with ground truth labels:
```json
{
  "description": "Test configuration for parking detection system evaluation",
  "version": "1.0",
  "tests": [
    {
      "test_id": "test_001",
      "source_file": "test_001.png",
      "description": "Test case description",
      "matrix": [
        ["P25:car", "P26:free", "P27:truck", "P28:free", "P29:car", "P30:bus"],
        ["P19:car", "P20:free", "P21:free", "P22:free", "P23:car", "P24:free"]
      ],
      "expected_occupied": 12,
      "expected_free": 18
    }
  ]
}
```

**Evaluation Output:**
- `matched_detections_*.csv`: Detailed ground truth vs prediction comparison
- `evaluation_metrics_*.json`: Complete metrics in JSON format
- `plots/confusion_matrix_*.png`: Visual confusion matrices
- `plots/performance_metrics_*.png`: Metrics comparison charts
- Console summary with accuracy, F1-scores, and per-class breakdown

### Detection Parameters
- `--conf`: Detection confidence threshold (default: 0.35)
- `--overlap_thr`: Minimum overlap to consider a space occupied (default: 0.15)
- `--history`: Temporal smoothing window size (default: 5 frames)
- `--weights`: YOLO model weights (default: yolo11x.pt, options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)

## Output Data

### Consolidated JSON Summary (`test/results/data/occupancy_summary.json`)
```json
{
  "parking_with_cars1.jpg": {
    "ts_utc": "2025-11-11T14:30:15.123456Z",
    "frame": 1,
    "spots": {
      "P1": {"status": "occupied", "overlap": 0.984},
      "P2": {"status": "free", "overlap": 0.0}
    },
    "unassigned_vehicles": [
      {
        "vehicle_id": 3,
        "vehicle_type": "car", 
        "status": "illegally_parked",
        "confidence": 0.78,
        "max_zone_overlap": 0.05
      }
    ]
  }
}
```

### Consolidated CSV Log (`test/results/data/occupancy_log.csv`)
Contains all results with source file identification:
- Source filename
- Timestamp (UTC)
- Frame number
- Spot ID or "UNASSIGNED"
- Occupancy status (free/occupied/illegally_parked)
- Vehicle type and confidence
- Overlap percentage
- Record type (parking_spot/unassigned_vehicle)

### Enhanced Visualizations
**Zone Visualization**:
- **Parking zones**: Green (free) → Dark green (occupied)
- **Traffic zones**: Orange (always visible for reference)
- **No-parking zones**: Red (violation detection areas)
- **Disabled zones**: Blue (accessibility compliant)

**Vehicle Status Indicators**:
- **Properly parked**: White boxes with [PARKED] label
- **Traffic vehicles**: Orange boxes with [TRAFFIC] label
- **Illegal parking**: Red boxes with [NO-PARK] or [ILLEGAL] labels
- **Partial parking**: Yellow boxes with [PARTIAL] label

**Information Overlay**:
- **Grid Mode**: Zone counts (Parking: X, Traffic: Y, No-parking: Z, Disabled: W)
- **Detection Mode**: Vehicle counts (Occupied, Free, Traffic, Partial, Illegal)
- **All images saved**: Automatic visualization saving with smart filenames

## Current Phase Status
**Phase 4 - Evaluation & Testing Framework Complete**

**Core System (Complete):**
- **Multi-zone architecture** with 4 zone types (P/T/N/D)
- **Advanced vehicle categorization** with 5 status types
- **Interactive zone mapping** with type selection and color coding
- **Accessibility compliance** with proper disabled parking colors
- **Modular code architecture** split into maintainable components
- **Grid analysis mode** for layout planning and zone counting
- **Enhanced visualization** with status-based colors and labels
- **Consolidated data logging** with source file identification

**Evaluation Framework (New - Complete):**
- **Ground truth testing** with JSON-based test configurations
- **Binary classification metrics** (Occupied vs Free detection)
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix with detailed breakdown
- **Multi-class metrics** (Vehicle type classification)
  - Per-class performance (Car, Truck, Bus, Motorcycle)
  - Macro and weighted averaging
  - Multi-class confusion matrix
- **Visual analytics dashboard**:
  - Confusion matrix heatmaps (binary & multi-class)
  - Performance metrics comparison charts
  - Per-class breakdown visualizations
  - Error analysis (TP/TN/FP/FN)
- **Comprehensive reporting**:
  - JSON metrics export
  - Matched detections CSV
  - Console summary with detailed statistics

**Recent Enhancements (Complete):**
- **Dual-mode main.py** with detection + editor modes
- **Smart directory reorganization** (data/ for inputs, test/ for results)
- **Cross-directory compatibility** (works from root or src)
- **Enhanced zone editor** with ID management and rename functionality
- **Smart save tracking** eliminates redundant save prompts
- **Dynamic path detection** for flexible execution
- **Always-save image results** with proper path handling
- **Comprehensive documentation** with UML diagrams and technical specs
- **Automated evaluation pipeline** with scikit-learn integration
- **Professional visualization** with seaborn heatmaps

**Next Phase Goals:**
- Real-time camera stream integration (RTSP/IP cameras)
- Database integration for persistent storage and analytics
- Web-based dashboard with live monitoring
- REST API development for external integrations
- Performance optimization for edge deployment
- Machine learning model fine-tuning for specific environments

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
