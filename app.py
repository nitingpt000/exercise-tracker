import streamlit as st
import cv2
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer

def main():
    st.title("Real-time Squat Form Analyzer")
    
    # Initialize pose detector and squat analyzer
    pose_detector = PoseDetector()
    squat_analyzer = SquatAnalyzer()
    
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Create placeholders for metrics
    rep_counter = st.empty()
    form_feedback = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            keypoints = pose_detector.detect_pose(frame)
            
            # Draw keypoints and connections
            frame = pose_detector.draw_keypoints(frame, keypoints)
            
            # Analyze squat
            reps, feedback = squat_analyzer.analyze_squat(keypoints)
            
            # Update metrics if available
            if reps is not None:
                rep_counter.markdown(f"## Reps: {reps}")
            if feedback:
                form_feedback.markdown(f"## Feedback: {feedback}")
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(frame_rgb, channels="RGB")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()