import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class PoseDetector:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = self.model.signatures['serving_default']

    def detect_pose(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad the image to keep aspect ratio
        img = tf.cast(tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192), dtype=tf.int32)
        
        # Detection
        results = self.movenet(img)
        keypoints = results['output_0'].numpy()[0][0]
        
        return keypoints

    def draw_keypoints(self, frame, keypoints):
        height, width, _ = frame.shape
        
        # Draw the detected points
        for idx, kp in enumerate(keypoints):
            y, x, score = kp
            if score > 0.3:  # Confidence threshold
                x = int(x * width)
                y = int(y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (5, 7), (7, 9),  # Right arm
            (6, 8), (8, 10),  # Left arm
            (5, 6), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15),  # Right leg
            (12, 14), (14, 16)   # Left leg
        ]
        
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if (keypoints[start_idx][2] > 0.3 and 
                keypoints[end_idx][2] > 0.3):
                
                start_point = (
                    int(keypoints[start_idx][1] * width),
                    int(keypoints[start_idx][0] * height)
                )
                end_point = (
                    int(keypoints[end_idx][1] * width),
                    int(keypoints[end_idx][0] * height)
                )
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame