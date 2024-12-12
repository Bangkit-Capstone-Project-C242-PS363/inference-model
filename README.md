# Enhanced Hand Gesture WebSocket Server API Documentation

## Overview

The Enhanced Hand Gesture WebSocket Server provides real-time hand gesture recognition through a WebSocket interface. It uses MediaPipe for hand detection and a custom TensorFlow model for gesture classification.

## Server Configuration

### Connection Details

- Protocol: WebSocket (ws://)
- Default Host: 0.0.0.0
- Default Port: 8080
- WebSocket URL: `wss://signmaster-inference-model-kji5w4ybbq-et.a.run.app`

### Dependencies

- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- SciPy
- websockets

## Communication Protocol

### Client to Server

Send base64-encoded image frames through the WebSocket connection.

```javascript
// Example client-side code
const ws = new WebSocket("ws://localhost:8080");
const base64Frame = ws.send(base64Frame); // your base64 encoded image
```

### Server to Client

The server responds with a JSON object containing:

```json
{
  "hand_detected": boolean,
  "predictions": [
    {
      "sign": string,
      "confidence": float
    }
  ],
  "landmarks": [
    {
      "x": float,
      "y": float,
      "z": float
    }
  ]
}
```

#### Response Fields

- `hand_detected`: Boolean indicating if a hand was detected
- `predictions`: Array of top prediction(s)
  - `sign`: Recognized gesture (0-9 or A-Z)
  - `confidence`: Confidence score (0-1)
- `landmarks`: Array of 21 hand landmarks with 3D coordinates (x, y, z)

## Gesture Recognition Details

### Supported Gestures

- Numbers: 0-9
- Letters: A-Z

### Feature Extraction

The server extracts several types of features:

- Basic landmark coordinates (63 features)
- Finger angles (5 features)
- Finger lengths (5 features)
- Palm features (3 features)
- Fingertip distances (10 features)

### Prediction Smoothing

- Temporal smoothing applied over last 5 frames
- Improves stability of predictions
- Maintains separate history per client

## Error Handling

### Connection Errors

- Server handles disconnections gracefully
- Cleans up client-specific resources on disconnect

### Processing Errors

- Invalid frames are handled without crashing
- Error messages are logged server-side

## Example Usage

```python
# Python client example using websockets
import asyncio
import websockets
import base64

async def send_frames():
    async with websockets.connect('ws://localhost:8080') as ws:
        # Read and encode your image
        with open('frame.jpg', 'rb') as img:
            base64_frame = base64.b64encode(img.read()).decode()

        # Send frame
        await ws.send(base64_frame)

        # Receive response
        response = await ws.recv()
        print(response)

asyncio.run(send_frames())
```

## Server Initialization

```python
server = EnhancedHandGestureWebSocketServer(
    model_path="mp_based.keras",
    scaler_path="feature_scaler.pkl",
    host="0.0.0.0",
    port=8080
)
```

### Parameters

- `model_path`: Path to the TensorFlow model file
- `scaler_path`: Path to the feature scaler pickle file
- `host`: Server host address
- `port`: Server port number
