# Real-Time Parking Occupancy Detection and Dynamic Pricing via Embedded Vision Systems
## Problem Statement
In dense urban areas, on-street parking demand exceeds supply, and current meter systems often fail to align paid time with actual use. This leads to low turnover, congestion from cruising for parking, revenue leakage, and high operating costs due to manual enforcement, while intrusive street hardware harms mobility and urban aesthetics. A computer-vision–based management system is needed to measure real-time occupancy per bay and charge for actual dwell without adding physical obstacles, while collecting anonymized usage data (occupancy and dwell) to generate heat maps by street, block, and time window. These heat maps enable detection of temporal spatial hotspots, targeted enforcement, evaluation of dynamic pricing, and evidence-based mobility planning. The objective is to increase turnover and availability, improve revenue efficiency, and reduce urban impact while ensuring privacy and regulatory compliance. Success will be measured by automatic identification of ≥90% of hotspots (top 10% occupancy) by time band and strict privacy compliance no storage of faces and plates, DPIA completed, and policies approved.

## Project Goals
- Optimize public parking availability through real-time detection and dynamic pricing.
- Reduce traffic congestion caused by drivers searching for parking.
- Eliminate the need for intrusive hardware by leveraging existing camera infrastructure.
- Enable data-driven urban policy via heatmaps and occupancy analytics.
- Improve revenue efficiency for municipalities through usage-based billing.
- Ensure privacy and scalability with anonymous tracking and edge-based processing.


## Tech Stack Overview
This system leverages edge computing and vision-based analytics to manage public parking dynamically and non-intrusively. The stack is composed of the following layers:
1. Hardware Layer
- Camera Input: Commercial RTSP-compatible cameras installed in public streets.
- Edge Devices: Embedded computers such as NVIDIA Jetson Orin or Raspberry Pi 4, responsible for local video processing.
2. Vision Computing Layer
- Detection Models: Real-time object detection using models like YOLOv8 or custom CNNs.
- Tracking Algorithms: Multi-object tracking with DeepSORT or similar frameworks.
- Inference Engine: Local logic to determine occupancy status, duration, and anonymized usage events.
3. Data Layer
- Event Format: Structured JSON records containing camera ID, timestamp, space ID, status, duration, and confidence score.
- Database: PostgreSQL or TimescaleDB for structured storage; optionally InfluxDB for time-series data.
4. API and Integration Layer
- REST API: Built with FastAPI or Flask to expose real-time data for operations and policy systems.
- Security: TLS encryption for data transmission; anonymized logging to preserve privacy.
5. Visualization and Analytics Layer
- Heatmaps: Generated using Mapbox, Leaflet.js, or D3.js to visualize occupancy patterns by time and location.
- Dashboard Tools: Optional integration with Grafana or custom web interfaces for monitoring and decision-making.


## Phase Status
Currently in Conception Phase
-  Abstract and system architecture defined
-  Initial UML diagrams drafted
-  Evaluating hardware options for edge deployment
-  Prototyping vision model for occupancy detection
-  Next: Simulated data ingestion and API scaffolding
