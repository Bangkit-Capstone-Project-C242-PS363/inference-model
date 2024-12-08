import asyncio
import websockets
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.spatial.distance import pdist
import json
import base64
tf.keras.config.enable_unsafe_deserialization()

class EnhancedHandGestureWebSocketServer:
    def __init__(self, model_path='mp_based.keras', scaler_path='feature_scaler.pkl', host='0.0.0.0', port=8080):
        """Initialize the WebSocket server with enhanced hand gesture recognition capabilities"""
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices for feature extraction
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_bases = [2, 5, 9, 13, 17]
        self.palm_landmarks = [0, 1, 5, 9, 13, 17]
        
        # Load model and scaler
        print("Loading model and preprocessing components...")
        self.model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Model and scaler loaded successfully!")
        
        # Server settings
        self.host = host
        self.port = port
        
        # Define class labels
        self.class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
        
        # Initialize prediction smoothing
        self.prediction_history = {}  # Dictionary to store history per client
        self.history_length = 5

    def calculate_finger_angles(self, landmarks_array):
        """Calculate angles between finger segments"""
        angles = []
        # Process each finger (except thumb)
        for finger_idx in range(1, 5):
            base = finger_idx * 4 + 1
            mid = finger_idx * 4 + 2
            tip = finger_idx * 4 + 3
            
            v1 = landmarks_array[mid] - landmarks_array[base]
            v2 = landmarks_array[tip] - landmarks_array[mid]
            
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            angles.append(angle)
        
        # Special case for thumb
        thumb_base = landmarks_array[1]
        thumb_mid = landmarks_array[2]
        thumb_tip = landmarks_array[4]
        
        v1 = thumb_mid - thumb_base
        v2 = thumb_tip - thumb_mid
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        thumb_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        angles.append(thumb_angle)
        
        return np.array(angles)

    def calculate_palm_features(self, landmarks_array):
        """Extract palm-specific geometric features"""
        palm_points = landmarks_array[self.palm_landmarks]
        hull = cv2.convexHull(palm_points[:, :2].astype(np.float32))
        palm_area = cv2.contourArea(hull)
        
        wrist_to_middle = landmarks_array[9] - landmarks_array[0]
        palm_angle = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
        
        palm_width = np.linalg.norm(landmarks_array[5] - landmarks_array[17])
        palm_height = np.linalg.norm(landmarks_array[0] - landmarks_array[9])
        palm_ratio = palm_width / palm_height if palm_height != 0 else 0
        
        return np.array([palm_area, palm_angle, palm_ratio])

    def calculate_fingertip_distances(self, landmarks_array):
        """Calculate pairwise distances between fingertips"""
        fingertip_positions = landmarks_array[self.finger_tips]
        distances = pdist(fingertip_positions)
        return distances

    def calculate_finger_lengths(self, landmarks_array):
        """Calculate normalized finger lengths"""
        lengths = []
        for finger_idx in range(5):
            if finger_idx == 0:
                base = 1
                tip = 4
            else:
                base = finger_idx * 4 + 1
                tip = finger_idx * 4 + 4
            
            length = np.linalg.norm(landmarks_array[tip] - landmarks_array[base])
            lengths.append(length)
        
        palm_size = np.linalg.norm(landmarks_array[0] - landmarks_array[5])
        normalized_lengths = np.array(lengths) / palm_size if palm_size != 0 else np.zeros(5)
        
        return normalized_lengths

    def extract_enhanced_features(self, frame):
        """Extract all features for gesture recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        basic_features = np.zeros(63)
        finger_angles = np.zeros(5)
        finger_lengths = np.zeros(5)
        palm_features = np.zeros(3)
        fingertip_distances = np.zeros(10)
        
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks_array = np.zeros((21, 3))
            for idx, landmark in enumerate(hand_landmarks.landmark):
                basic_features[idx*3:(idx*3)+3] = [landmark.x, landmark.y, landmark.z]
                landmarks_array[idx] = [landmark.x, landmark.y, landmark.z]
                landmarks_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            finger_angles = self.calculate_finger_angles(landmarks_array)
            finger_lengths = self.calculate_finger_lengths(landmarks_array)
            palm_features = self.calculate_palm_features(landmarks_array)
            fingertip_distances = self.calculate_fingertip_distances(landmarks_array)
            
            all_features = np.concatenate([
                basic_features,
                finger_angles,
                finger_lengths,
                palm_features,
                fingertip_distances
            ])
            
            return all_features, landmarks_list, True
            
        return np.zeros(86), landmarks_list, False

    def smooth_predictions(self, client_id, new_prediction):
        """Apply temporal smoothing to predictions"""
        if client_id not in self.prediction_history:
            self.prediction_history[client_id] = []
            
        self.prediction_history[client_id].append(new_prediction)
        if len(self.prediction_history[client_id]) > self.history_length:
            self.prediction_history[client_id].pop(0)
        return np.mean(self.prediction_history[client_id], axis=0)

    async def process_frame(self, frame_data, client_id):
        """Process a single frame and return inference results"""
        # Convert base64 image to numpy array
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        response = {
            'hand_detected': False,
            'predictions': [],
            'landmarks': None
        }
        
        features, landmarks, hand_detected = self.extract_enhanced_features(frame)
        
        if hand_detected:
            response['hand_detected'] = True
            response['landmarks'] = landmarks
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predictions = self.model.predict(features_scaled, verbose=0)[0]
            smoothed_predictions = self.smooth_predictions(client_id, predictions)
            
            top_3_idx = np.argsort(smoothed_predictions)[-3:][::-1]
            response['predictions'] = [
                {
                    'sign': self.class_names[idx],
                    'confidence': float(smoothed_predictions[idx])
                }
                for idx in top_3_idx
            ]
        
        return json.dumps(response)

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        print(f"Client connected from {websocket.remote_address}")
        try:
            async for message in websocket:
                result = await self.process_frame(message, client_id)
                await websocket.send(result)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
            if client_id in self.prediction_history:
                del self.prediction_history[client_id]
        except Exception as e:
            print(f"Error processing client request: {str(e)}")

    async def start_server(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Server started at ws://{self.host}:{self.port}")
            await asyncio.Future()

def main():
    server = EnhancedHandGestureWebSocketServer()
    asyncio.run(server.start_server())

if __name__ == "__main__":
    main()