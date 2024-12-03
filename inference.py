import cv2
import numpy as np
import tensorflow as tf

class ASLInference:
    def __init__(self, model_path='fine_tuned_model5.keras'):

        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        self.class_names = sorted([
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ])
        
        
        # Verify number of classes matches model output
        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError(f"Model expects {self.model.output_shape[-1]} classes but {len(self.class_names)} classes provided")
        
    def preprocess_frame(self, frame):

        # Resize to match model's expected input size
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB (important as training data was in RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb / 255.0
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def run_inference(self):
        print("\nStarting inference... Press 'q' to quit.")
        print("\nTop 3 predictions will be shown for each frame.")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Get the region of interest (ROI)
            height, width = frame.shape[:2]
            size = min(height, width)
            x = (width - size) // 2
            y = (height - size) // 2
            roi = frame[y:y+size, x:x+size]
            
            # Draw ROI rectangle
            cv2.rectangle(display_frame, (x, y), (x+size, y+size), (0, 255, 0), 2)
            
            # Preprocess ROI
            input_tensor = self.preprocess_frame(roi)
            
            # Run inference
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                (self.class_names[idx], predictions[idx])
                for idx in top_3_idx
            ]

            y_text = 30
            for i, (sign, conf) in enumerate(top_3_predictions):
                text = f"#{i+1}: {sign} ({conf:.2f})"
                cv2.putText(display_frame, text, (10, y_text), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_text += 30
                
            cv2.imshow('ASL Sign Language Detection', display_frame)

            print("\r", end="")
            for i, (sign, conf) in enumerate(top_3_predictions):
                print(f"#{i+1}: {sign} ({conf:.2f})  ", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asl_detector = ASLInference('model_alphabetnumerik_sebelum_hyperparametertuning.keras')
    asl_detector.run_inference()