import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from PIL import Image
import mediapipe as mp
# pip install websockets pillow tensorflow opencv-python mediapipe==0.10.3 numpy
class ASLWebSocketClient:
    def __init__(self, server_url='ws://localhost:8765'):
        """Initialize the WebSocket client with video capture capabilities"""
        self.server_url = server_url
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        # Initialize MediaPipe drawing utilities for visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
    async def send_frame(self, websocket, frame):
        """Encode and send a frame to the server"""
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame to server
        await websocket.send(frame_data)
        
        # Receive and parse response
        response = await websocket.recv()
        # print(response)
        return json.loads(response)

    def draw_predictions(self, frame, predictions):
        """Draw prediction results on the frame"""
        y_text = 30
        for i, pred in enumerate(predictions):
            text = f"#{i+1}: {pred['sign']} ({pred['confidence']:.2f})"
            cv2.putText(frame, text, (10, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_text += 30

    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on the frame"""
        h, w = frame.shape[:2]
        connections = self.mp_hands.HAND_CONNECTIONS
        
        # Convert landmarks to MediaPipe format for drawing
        mp_landmarks = []
        for landmark in landmarks:
            mp_landmarks.append(
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2
                )
            )
            
        # Draw landmarks and connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            start_pos = (int(start_point['x'] * w), int(start_point['y'] * h))
            end_pos = (int(end_point['x'] * w), int(end_point['y'] * h))
            
            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
            
        for landmark in landmarks:
            pos = (int(landmark['x'] * w), int(landmark['y'] * h))
            cv2.circle(frame, pos, 3, (0, 255, 0), -1)

    async def run(self):
        """Run the client, capturing and processing frames"""
        print(f"\nConnecting to server at {self.server_url}")
        print("Press 'q' to quit")
        
        async with websockets.connect(self.server_url) as websocket:
            print("Connected to server!")
            
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame with server
                try:
                    result = await self.send_frame(websocket, frame)
                    
                    # Display results
                    if result['hand_detected']:
                        # Draw landmarks if available
                        if result['landmarks']:
                            self.draw_landmarks(frame, result['landmarks'])
                        
                        # Draw predictions
                        self.draw_predictions(frame, result['predictions'])
                    else:
                        cv2.putText(frame, "No hand detected", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
                
                # Display frame
                cv2.imshow('ASL Sign Language Detection', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete!")

def main():
    client = ASLWebSocketClient()
    asyncio.get_event_loop().run_until_complete(client.run())

if __name__ == "__main__":
    main()