# 🚗 Parking Compliance Advisor

A historical, probabilistic parking analysis tool for the San Francisco Bay Area. This Streamlit web application helps drivers find **legal parking** by analyzing enforcement patterns and complaint hotspots.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ⚠️ Important Disclaimer

- ❌ This app does **NOT** track real-time enforcement vehicle locations
- ❌ This app does **NOT** suggest or encourage illegal parking
- ✅ All data is historical and for informational purposes only
- ✅ Always park legally and check posted signs

## Features

### 📊 Data Analysis
- Load and process SF parking citation data
- Load and process San Jose 311 illegal parking complaints
- Time-based filtering (hour of day, day of week)

### 🗺️ Interactive Map
- SF enforcement activity heatmap (red)
- SJ complaint hotspot heatmap (blue)
- Risk grid overlay with color-coded cells
- Toggle layers on/off

### 📈 Risk Scoring
- ~200m grid cell resolution
- Exponential decay weighting (configurable halflife)
- Combined risk score (SF: 70%, SJ: 30% by default)
- Normalized scores (0-1 range)

### 📍 Destination Lookup
- Enter coordinates to check risk score
- Find nearby lower-risk areas for legal parking
- Distance-sorted suggestions

### 📥 Data Export
- Download combined risk grid CSV
- Download filtered SF tickets CSV

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nucleushuy/TRACK_PARK_COMPLIANTS.git
   cd TRACK_PARK_COMPLIANTS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   
   Navigate to `http://localhost:8501`

## Data Format

### SF Tickets (`sf_tickets_last30.csv`)
```csv
timestamp,latitude,longitude
2026-01-15 08:30:00,37.7749,-122.4194
```

### SJ Complaints (`sj_illegal_parking_last30.csv`)
```csv
timestamp,latitude,longitude
2026-01-15 10:00:00,37.3382,-121.8863
```

**Required columns:**
- `timestamp` - DateTime in any parseable format
- `latitude` - Decimal degrees (Bay Area: 36.5 to 38.5)
- `longitude` - Decimal degrees (Bay Area: -123.0 to -121.0)

## Configuration

Key parameters can be adjusted in the sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Hour Range | 0-23 | Filter by time of day |
| Days of Week | All | Filter by weekday |
| Recency Halflife | 7 days | Decay rate for older events |
| SF Weight | 0.7 | Weight for ticket data |
| SJ Weight | 0.3 | Weight for complaint data |

## Project Structure

```
TRACK_PARK_COMPLIANTS/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── sf_tickets_last30.csv       # SF parking citation data
├── sj_illegal_parking_last30.csv  # SJ 311 complaint data
└── README.md                   # This file
```

## Technical Details

### Risk Scoring Algorithm

1. **Grid Assignment**: Each event is assigned to a ~200m grid cell
2. **Recency Weighting**: `weight = 0.5^(days_ago / halflife)`
3. **Aggregation**: Sum of weighted events per cell
4. **Normalization**: Scale to [0, 1] range
5. **Combination**: `combined = (SF_weight × SF_score) + (SJ_weight × SJ_score)`

### Coordinate Validation

All coordinates are validated against Bay Area bounds:
- Latitude: 36.5° to 38.5°
- Longitude: -123.0° to -121.0°

## Dependencies

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `pydeck>=0.8.0` - Map visualization (deck.gl)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- San Francisco Open Data Portal
- San Jose 311 Service
- Streamlit team for the excellent framework
- deck.gl for powerful map visualizations

---

**Remember:** Always park legally and check posted signs. This tool provides historical analysis only.
