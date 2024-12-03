import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

class ASLInference:
    def __init__(self, model_path='asl_model.keras'):
        """
        Initialize ASL inference system with hand tracking
        """
        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
            
        # Class mapping
        self.class_names = sorted([
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ])
        
    def get_hand_bbox(self, hand_landmarks, image_shape):
        """
        Get bounding box coordinates from hand landmarks
        """
        if hand_landmarks is None:
            return None
        
        h, w = image_shape[:2]
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        # Add padding around the hand
        padding = 50
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        # Ensure square bbox by taking max of width and height
        bbox_size = max(x_max - x_min, y_max - y_min)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        
        # Adjust bbox to be square
        x_min = max(0, x_center - bbox_size // 2)
        x_max = min(w, x_center + bbox_size // 2)
        y_min = max(0, y_center - bbox_size // 2)
        y_max = min(h, y_center + bbox_size // 2)
        
        return (x_min, y_min, x_max, y_max)
        
    def preprocess_frame(self, frame):
        """
        Preprocess the frame exactly as done during training
        """
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def run_inference(self):
        """Run continuous inference on webcam feed with hand tracking"""
        print("\nStarting inference... Press 'q' to quit.")
        print("\nTop 3 predictions will be shown for each frame.")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get bounding box
                bbox = self.get_hand_bbox(hand_landmarks, frame.shape)
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Extract hand ROI
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    if hand_roi.size != 0:  # Check if ROI is not empty
                        # Preprocess ROI
                        input_tensor = self.preprocess_frame(hand_roi)
                        
                        # Run inference
                        predictions = self.model.predict(input_tensor, verbose=0)[0]
                        
                        # Get top 3 predictions
                        top_3_idx = np.argsort(predictions)[-3:][::-1]
                        top_3_predictions = [
                            (self.class_names[idx], predictions[idx])
                            for idx in top_3_idx
                        ]
                        
                        # Display results
                        y_text = 30
                        for i, (sign, conf) in enumerate(top_3_predictions):
                            text = f"#{i+1}: {sign} ({conf:.2f})"
                            cv2.putText(display_frame, text, (10, y_text), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            y_text += 30
            else:
                # Display "No hand detected" when no hand is visible
                cv2.putText(display_frame, "No hand detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Display frame
            cv2.imshow('ASL Sign Language Detection', display_frame)
            
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

if __name__ == "__main__":
    try:
        asl_detector = ASLInference('asl_model.keras')
        asl_detector.run_inference()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        asl_detector.cleanup()
    except Exception as e:
        print(f"Program error: {str(e)}")