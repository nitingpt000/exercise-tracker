import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class PoseDetector:
    def __init__(self, model_variant='thunder', smoothing_factor=0.7):
        """
        Initialize the pose detector.
        
        Args:
            model_variant (str): Choose 'thunder' for improved accuracy or 'lightning' for faster inference.
            smoothing_factor (float): Factor for exponential moving average smoothing (0=no smoothing, 1=full smoothing).
        """
        if model_variant == 'thunder':
            self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
        else:
            self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = self.model.signatures['serving_default']
        self.smoothing_factor = smoothing_factor
        self.prev_keypoints = None

    def detect_pose(self, frame):
        """
        Detect pose keypoints in the input frame and apply temporal smoothing.
        
        Args:
            frame (numpy.ndarray): Input image frame in BGR format.
        
        Returns:
            numpy.ndarray: Array of keypoints with shape [17, 3] (y, x, confidence).
        """
        # Convert frame to RGB and resize with padding to 192x192 (expected by MoveNet)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run model inference
        results = self.movenet(input_image)
        keypoints = results['output_0'].numpy()[0][0]
        
        # Apply exponential moving average smoothing to reduce jitter
        if self.prev_keypoints is not None:
            keypoints = self.smoothing_factor * self.prev_keypoints + (1 - self.smoothing_factor) * keypoints
        self.prev_keypoints = keypoints.copy()
        
        return keypoints

    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.5):
        """
        Draw detected keypoints and skeletal connections on the frame.
        
        Args:
            frame (numpy.ndarray): The image frame on which to draw.
            keypoints (numpy.ndarray): Array of keypoints [17, 3].
            confidence_threshold (float): Minimum confidence required to display a keypoint.
        
        Returns:
            numpy.ndarray: Frame with keypoints and connections drawn.
        """
        height, width, _ = frame.shape
        
        # Draw individual keypoints
        for idx, kp in enumerate(keypoints):
            y, x, score = kp
            if score > confidence_threshold:
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # Define skeletal connections
        connections = [
            (5, 7), (7, 9),    # Right arm
            (6, 8), (8, 10),   # Left arm
            (5, 6), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), # Right leg
            (12, 14), (14, 16)  # Left leg
        ]
        
        # Draw lines between connected keypoints
        for start_idx, end_idx in connections:
            if (keypoints[start_idx][2] > confidence_threshold and
                keypoints[end_idx][2] > confidence_threshold):
                start_point = (int(keypoints[start_idx][1] * width), int(keypoints[start_idx][0] * height))
                end_point = (int(keypoints[end_idx][1] * width), int(keypoints[end_idx][0] * height))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame


# wider body frame
# accurate key poiints
#workout with rest and 10 exercises