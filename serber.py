import asyncio
import websockets
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import base64
from PIL import Image
import io

class ASLWebSocketServer:
    def __init__(self, model_path='asl_model.keras', host='0.0.0.0', port=8765):
        """Initialize the WebSocket server with ASL inference capabilities"""
        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Server settings
        self.host = host
        self.port = port
        
        # Class mapping
        self.class_names = sorted([
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ])

    def get_hand_bbox(self, hand_landmarks, image_shape):
        """Get bounding box coordinates from hand landmarks"""
        if hand_landmarks is None:
            return None
        
        h, w = image_shape[:2]
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        padding = 50
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        bbox_size = max(x_max - x_min, y_max - y_min)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        
        x_min = max(0, x_center - bbox_size // 2)
        x_max = min(w, x_center + bbox_size // 2)
        y_min = max(0, y_center - bbox_size // 2)
        y_max = min(h, y_center + bbox_size // 2)
        
        return (x_min, y_min, x_max, y_max)

    def preprocess_frame(self, frame):
        """Preprocess the frame for model inference"""
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched

    async def process_frame(self, frame_data):
        """Process a single frame and return inference results"""
        # Convert base64 image to numpy array
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        response = {
            'hand_detected': False,
            'predictions': [],
            'landmarks': None
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get hand landmarks for client-side visualization
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            response['landmarks'] = landmarks
            
            # Get bounding box and run inference
            bbox = self.get_hand_bbox(hand_landmarks, frame.shape)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                hand_roi = frame[y_min:y_max, x_min:x_max]
                
                if hand_roi.size != 0:
                    input_tensor = self.preprocess_frame(hand_roi)
                    predictions = self.model.predict(input_tensor, verbose=0)[0]
                    
                    # Get top 3 predictions
                    top_3_idx = np.argsort(predictions)[-3:][::-1]
                    response['predictions'] = [
                        {
                            'sign': self.class_names[idx],
                            'confidence': float(predictions[idx])
                        }
                        for idx in top_3_idx
                    ]
                    response['hand_detected'] = True
        
        return json.dumps(response)

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        print(f"Client connected from {websocket.remote_address}")
        try:
            async for message in websocket:
                print(f"Received message from {websocket.remote_address}")
                result = await self.process_frame(message)
                await websocket.send(result)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Error processing client request: {str(e)}")

    async def start_server(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

        

def main():
    server = ASLWebSocketServer()
    asyncio.run(server.start_server())

if __name__ == "__main__":
    main()