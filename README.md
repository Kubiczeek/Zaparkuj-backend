# Zaparkuj server side

A FastAPI-based backend service for monitoring and predicting parking lot occupancy using computer vision.

## Features

- Real-time vehicle counting from IP camera feeds using YOLOv8
- Historical parking data collection
- Occupancy prediction based on historical patterns
- REST API for accessing parking information
- Authentication for secure access
- Export of parking data in CSV format

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster vehicle detection)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Zaparkuj-backend.git
cd Zaparkuj-backend
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure parking lots in `parking-lots/data.json`

4. Start the server:

```bash
uvicorn main:app --reload
```

## Configuration

Parking lots are configured in the `parking-lots/data.json` file with the following structure:

```json
[
  {
    "id": 1,
    "url": "http://camera-url/snap.jpg",
    "name": "Parking Name",
    "maxParkingSpaces": 17,
    "maxHandicappedSpaces": 2,
    "paidTime": "Po-Pá: 7-17",
    "prices": [
      {"time": "Prvních 30min", "price": 3},
      {"time": "Každá další hodina", "price": 15}
    ],
    "center": [49.608902, 15.579536],
    "polygon": [
      [49.608989, 15.579777],
      [49.608791, 15.579724],
      [49.608815, 15.579332],
      [49.609031, 15.579370]
    ]
  }
]
```

## API Endpoints

### Base URL

`/api/v1`

### Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | API welcome message | None |
| `/parking-occupancy/{id}` | GET | Get current occupancy for a parking lot | `id`: Parking lot ID, `conf`: Confidence threshold (0.0-1.0) |
| `/parking-occupancy/{id}/history` | GET | Get predicted occupancy | `id`: Parking lot ID, `time_arrival`: Optional time in HH:MM format |
| `/admin/csv` | GET | Export database as CSV | None |

## Authentication

The API uses Basic Authentication. Credentials must be provided with each request.

## Vehicle Detection

The system uses YOLOv8 for vehicle detection. It can detect cars, motorcycles, buses, and trucks from camera feeds. The detection confidence threshold can be adjusted via the API.

## How It Works

1. The system periodically fetches images from configured IP cameras
2. Vehicle detection is performed on each image
3. Occupancy data is stored in a database
4. Historical patterns are analyzed to predict future occupancy
5. All data is accessible via the REST API

## License

[Include your license information here]